---
title: "Round 17 report: the information matrix in absolute terms, and the cost of the minimum distance (Parts A and B)"
date: "2026-11-11"
---

English copy of `report_round17_CN.md`. Where the two differ, the Chinese version is the report of record.

**Prompt.** `reports_CN/prompt_round17_absolute_diagnostics_EN.md`. Branch `round17-absolute-diagnostics`, created from the head of `round14-one-parameter-rule` at **`6b79535`** (the Round 14 report). Commits:
- `1b426b0`: the prompt (first commit);
- `b403c57`: `round17/predictions_round17.md`, before any computation (second commit);
- `4f75460`: code;
- `8a845c9`: Part A;
- `68956af`: Part B;
- `d6784e3`: summary;
- this report: the commit after that.

Part C was not run (on hold).

**Scope.**
- Input U, budget B = 316.8921168493721. No DP.
- Part A uses the σ/24 matrix of each design. The fixed residual threshold and class W are matrices of round 10's model.
- μ are the generalised eigenvalues of J v = μ J_full v, i.e. the fraction of the full-data information kept in that direction. cond is the 2-norm condition number.
- No causes are claimed.

**Sources.** Paths are relative to `code_8_3_stage2/stage2_pkg/evaluator/round17/`.
- `[L X:n]` is line n of `logs/X.log`; `[J a.b]` is field `a.b` of `results/round17.json`.
- Code: `diag17.py` (diagnostics, rule with minimum half-width, fit, continuous Monte Carlo), `run_round17.py` (stages `partA`, `partB`, `summary`), `summary17.py`.
- The J matrices are in `results/partA/` and `results/partB/`.

## 1. Answers to Q1–Q4

Sources: [L summary:16–20]; [J questions].

- **Q1.** Fraction of the full-data information kept in the worst and best direction (μ_min / μ_max), and cond relative to full data (98.679):
  - constant SoD 0.049 / 0.165, cond −3.8 %;
  - round-4 table 0.188 / 0.238, −10.0 %;
  - T₁ 0.246 / 0.308, −9.4 %;
  - T′ 0.239 / 0.321, −9.8 %;
  - rule R(37, 13) 0.263 / 0.290, +0.45 %;
  - fixed residual threshold 0.327 / 0.329, −0.09 %;
  - class-W optimum 0.308 / 0.372, −10.3 %.
- **Q2 — larger in the worst direction, for every design.** μ_min(design)/μ_min(SoD) against the ratio of geometric means exp((I − I_SoD)/9):

  | design | μ_min ratio | geometric-mean ratio | worst / average |
  |---|---:|---:|---:|
  | round-4 table | 3.87 | 2.39 | ×1.62 |
  | T₁ | 5.07 | 3.09 | ×1.64 |
  | T′ | 4.93 | 3.19 | ×1.55 |
  | rule | 5.41 | 3.14 | ×1.72 |
  | fixed threshold | 6.74 | 3.69 | ×1.83 |
  | class W | 6.34 | 3.74 | ×1.70 |
- **Q3.** Among the designed triggers, only the rule has its condition number within 5 % of the full-data value (99.13, +0.45 %). The three tables lie about 10 % **below** it: round-4 88.78, T₁ 89.42, T′ 89.00. For reference, constant SoD is 94.92 (−3.8 %), the fixed threshold 98.59 (−0.09 %) and class W 88.51 (−10.3 %).
- **Q4 — yes, monotone on σ/24, σ/120 and in the continuous Monte Carlo.**

  | | σ/24 | σ/120 | Monte Carlo |
  |---|---:|---:|---:|
  | I(0.25 σ) − I(1 σ) | 0.0966 | 0.0965 | 0.1033 |
  | I(1 σ) − I(2 σ) | 0.1894 | 0.1892 | 0.1877 |

## 2. Part A — diagnostics per design

Each design reproduced its archived log det J with relative deviation 0 before its diagnostics were used [L partA:2–28]. S = Σ W_k.

| design | log det J | ΔI over SoD | I − I_full | E[N] | S | μ_min | μ_max | μ_max/μ_min | cond (vs full) | SD ratio min / geo / max | tr(J⁻¹)/tr(J_full⁻¹) | λ_min/λ_min,full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| full data | 219.0085 | +21.808 | 0 | 5000 | 5000 | 1 | 1 | 1 | 98.679 | 1 / 1 / 1 | 1 | 1 |
| constant SoD at Δ\* | 197.2010 | 0 | −21.808 | 316.89 | 466.16 | 0.0485 | 0.1655 | 3.410 | 94.922 (−3.8 %) | 3.108 / 3.359 / 3.922 | 10.648 | 0.098 |
| round-4 table | 205.0314 | +7.830 | −13.977 | 316.60 | 1037.26 | 0.1876 | 0.2383 | 1.271 | 88.782 (−10.0 %) | 2.070 / 2.174 / 2.222 | 4.413 | 0.231 |
| T₁ (100 actions) | 207.3552 | +10.154 | −11.653 | 316.73 | 1344.21 | 0.2458 | 0.3081 | 1.253 | 89.420 (−9.4 %) | 1.821 / 1.911 / 1.969 | 3.409 | 0.297 |
| T′ (884 actions) | 207.6467 | +10.446 | −11.362 | 316.84 | 1384.43 | 0.2393 | 0.3209 | 1.341 | 89.000 (−9.8 %) | 1.787 / 1.880 / 1.942 | 3.296 | 0.308 |
| rule R(37, 13) | 207.5003 | +10.299 | −11.508 | 316.73 | 1392.84 | 0.2626 | 0.2898 | 1.104 | 99.127 (+0.45 %) | 1.881 / 1.895 / 1.904 | 3.602 | 0.277 |
| fixed residual threshold | 208.9566 | +11.756 | −10.052 | 316.89 | 1636.15 | 0.3271 | 0.3289 | 1.006 | 98.595 (−0.09 %) | 1.746 / 1.748 / 1.748 | 3.053 | 0.328 |
| class-W optimum | 209.0685 | +11.868 | −9.940 | 316.89 | 1619.45 | 0.3077 | 0.3723 | 1.210 | 88.512 (−10.3 %) | 1.650 / 1.737 / 1.771 | 2.806 | 0.362 |

**The nine μ of every design** [J table_A.\*.mu]:

| design | μ₁ | μ₂ | μ₃ | μ₄ | μ₅ | μ₆ | μ₇ | μ₈ | μ₉ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full data | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| constant SoD at Δ* | 0.0485 | 0.0551 | 0.0815 | 0.0858 | 0.0930 | 0.0978 | 0.1079 | 0.1114 | 0.1655 |
| round-4 table | 0.1876 | 0.1962 | 0.1969 | 0.2058 | 0.2110 | 0.2162 | 0.2227 | 0.2356 | 0.2383 |
| T₁ | 0.2458 | 0.2485 | 0.2605 | 0.2620 | 0.2682 | 0.2784 | 0.3006 | 0.3015 | 0.3081 |
| T′ | 0.2393 | 0.2645 | 0.2656 | 0.2742 | 0.2810 | 0.2943 | 0.3015 | 0.3154 | 0.3209 |
| rule R(37, 13) | 0.2626 | 0.2743 | 0.2751 | 0.2768 | 0.2782 | 0.2798 | 0.2820 | 0.2881 | 0.2898 |
| fixed residual threshold | 0.3271 | 0.3271 | 0.3271 | 0.3271 | 0.3271 | 0.3271 | 0.3271 | 0.3271 | 0.3289 |
| class-W optimum | 0.3077 | 0.3124 | 0.3145 | 0.3194 | 0.3218 | 0.3327 | 0.3406 | 0.3677 | 0.3723 |

**Per-parameter Cramér–Rao standard-deviation ratio to full data**, sqrt(diag(J⁻¹)/diag(J_full⁻¹)), in the parameter order of θ [J table_A.\*.sd_ratio]:

| design | G(0) | Re G(jω₁) | Im G(jω₁) | Re G(jω₂) | Im G(jω₂) | Re G(jω₃) | Im G(jω₃) | Re G(jω₄) | Im G(jω₄) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full data | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| constant SoD at Δ* | 3.333 | 3.922 | 3.677 | 3.329 | 3.327 | 3.108 | 3.240 | 3.154 | 3.327 |
| round-4 table | 2.222 | 2.199 | 2.217 | 2.213 | 2.215 | 2.071 | 2.082 | 2.070 | 2.099 |
| T₁ | 1.938 | 1.969 | 1.930 | 1.937 | 1.938 | 1.830 | 1.821 | 1.830 | 1.833 |
| T′ | 1.922 | 1.876 | 1.942 | 1.913 | 1.923 | 1.803 | 1.787 | 1.800 | 1.800 |
| rule R(37, 13) | 1.896 | 1.881 | 1.904 | 1.891 | 1.898 | 1.900 | 1.898 | 1.903 | 1.894 |
| fixed residual threshold | 1.748 | 1.748 | 1.748 | 1.747 | 1.748 | 1.748 | 1.747 | 1.748 | 1.746 |
| class-W optimum | 1.771 | 1.761 | 1.768 | 1.768 | 1.771 | 1.662 | 1.650 | 1.659 | 1.662 |

The outside reference value for the rule (from its continuous-rule Monte Carlo: μ between 0.262 and 0.290, cond ≈ 99.2) lies close to the lattice σ/24 values here (0.2626–0.2898, 99.13).

## 3. Part B — minimum distance M_min from the last transmitted value

The rule is L_i = max(M_min, i − FL[p] + n_p), U_i = max(M_min, CL[p] − i + n_p), fitted with Round 14's `fit_rule`. The Monte Carlo uses the continuous rule with r-side distance m = M_min/24 (R = 10⁴, Round 14's seeds) [L partB:10–12, 21–23, 32–34, 44–46, 56–58, 68–70]; [J table_B].

| M_min (m) | rule | e_eff | E[N] σ/24 | I σ/24 | E[N] σ/120 | I σ/120 | μ_min | μ_max | cond | SD ratio min–max | MC e_fit | MC I at B | outside e_fit / I |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 6 (0.25 σ) | R(38, 19) | 1.6062 | 316.303 | 207.5969 | 316.073 | 207.5943 | 0.2572 | 0.3003 | 99.259 | 1.864–1.902 | 1.6051 | 207.6117 | 1.607 / 207.584 |
| 12 (0.5 σ) | R(37, 3) | 1.5979 | 316.061 | 207.5604 | 315.832 | 207.5579 | 0.2587 | 0.2967 | 99.225 | 1.870–1.903 | 1.5964 | 207.5789 | 1.598 / 207.553 |
| 18 (0.75 σ) | R(37, 8) | 1.5875 | 316.719 | 207.5375 | 316.491 | 207.5350 | 0.2608 | 0.2935 | 99.183 | 1.875–1.903 | 1.5867 | 207.5460 | 1.588 / 207.522 |
| 24 (1 σ) | R(37, 13) | 1.5771 | 316.725 | 207.5003 | 316.499 | 207.4978 | 0.2626 | 0.2898 | 99.127 | 1.881–1.904 | 1.5763 | 207.5084 | 1.578 / 207.490 |
| 36 (1.5 σ) | R(36, 4) | 1.5542 | 316.724 | 207.4172 | 316.502 | 207.4148 | 0.2662 | 0.2824 | 98.941 | 1.896–1.909 | 1.5536 | 207.4241 | 1.554 / 207.416 |
| 48 (2 σ) | R(36, 17) | 1.5271 | 316.638 | 207.3109 | 316.423 | 207.3087 | 0.2708 | 0.2746 | 98.596 | 1.913–1.917 | 1.5262 | 207.3207 | 1.528 / 207.298 |

**Steps of I as M_min decreases** (nat):

| | 48→36 | 36→24 | 24→18 | 18→12 | 12→6 |
|---|---:|---:|---:|---:|---:|
| σ/24 | +0.106 | +0.083 | +0.037 | +0.023 | +0.036 |
| σ/120 | +0.106 | +0.083 | +0.037 | +0.023 | +0.036 |
| Monte Carlo | +0.103 | +0.084 | +0.038 | +0.033 | +0.033 |

**Differences:**
- I(0.25 σ) − I(1 σ) = 0.0966 (σ/24), 0.0965 (σ/120), 0.1033 (Monte Carlo).
- I(1 σ) − I(2 σ) = 0.1894, 0.1892, 0.1877.

As M_min decreases, μ_min decreases slightly (0.2708 → 0.2572), μ_max increases (0.2746 → 0.3003) and cond increases from 98.60 to 99.26.

**The Monte Carlo against the outside run** (4000 trajectories, another seed; not a gate):
- e_fit differs by −0.0004 … −0.0019.
- log det J at B is higher here by +0.008 … +0.028.
- At m = 1 the value here (207.5084) agrees with Round 14's continuous-rule interpolation (207.5082).

Each scan needed two Monte Carlo runs: one at the lattice e_eff, with E[N] just below B, and one 0.04 σ lower, above B.

## 4. Predictions (`predictions_round17.md`, `b403c57`)

**4 hits, 1 miss** [L summary:25–30].

| line | values | result |
|---|---|---|
| All reproductions pass | every log det J at rel 0; references ≤ 7.5e-16; identity ≤ 3.5e-15; M_min = 24 bitwise; round 10 bitwise | hit |
| Q1/Q3: rule μ ∈ [0.255, 0.270] / [0.283, 0.297], cond ∈ [97, 101]; T′ μ_min ≥ 0.26, μ_max ≤ 0.31, cond ∈ [97, 101]; fixed threshold all μ within 0.001 of 0.327; class W all μ in [0.30, 0.37] | rule 0.2626 / 0.2898 / 99.13 ✓; **T′ 0.2393 / 0.3209 / 89.00 ✗**; **fixed threshold: eight μ = 0.3271, one 0.3289 (0.0019 off) ✗**; **class W μ_max 0.3723 ✗** | **miss** |
| Q2: μ_min ratio ≥ 5; geometric-mean ratio about 3.1 | 5.41; 3.14 | hit |
| SD ratios: rule all in [1.80, 2.00]; SoD all in [2.4, 4.6] | 1.881–1.904; 3.108–3.922 | hit |
| Q4: monotone; I(0.25) − I(1) ∈ [0.05, 0.15]; I(1) − I(2) ∈ [0.15, 0.25] (σ/24, σ/120, MC) | monotone on all three; 0.097 / 0.096 / 0.103; 0.189 / 0.189 / 0.188 | hit |

## 5. Checks (prompt §4)

1. **Reproductions** [L partA:2–32]:
   ```
   [L partA:2]  REPRO full (full data): log det J 219.00850592919505 vs 219.00850592919505 [prompt §1] rel 0.0e+00 -> PASS
   [L partA:5]  SOD archived J (round4/results/J_U_SoD.npy) vs accepted operator ...: {'bitwise': True, 'max_rel': 0.0}
   [L partA:9]  REPRO round4 (...): log det J 205.03141309913676 vs 205.03141309913676 ... rel 0.0e+00 -> PASS
   [L partA:12] REPRO T1 (...): log det J 207.35521730257756 vs 207.35521730257756 ... rel 0.0e+00 -> PASS
   [L partA:15] TPRIME action set S u S' (884 actions); edge nodes LN/UN equal to the archived npz: True
   [L partA:16] REPRO Tprime (...): log det J 207.64665023111849 vs 207.64665023111849 ... rel 0.0e+00 -> PASS
   [L partA:19] REPRO rule (...): log det J 207.5002944958199 vs 207.5002944958199 ... rel 0.0e+00 -> PASS
   [L partA:22] REPRO fixed (...): log det J 208.95657763962944 vs 208.95657763962944 ... rel 0.0e+00 -> PASS
   [L partA:26] REPRO classW (...): log det J 209.0685398686887 vs 209.0685398686887 ... rel 0.0e+00 -> PASS
   [L partA:29] REF cond_full: 98.67923064906215 vs 98.67923064906208 rel 7.2e-16 -> PASS
   [L partA:30] REF sod_mu_min: 0.048517708610531125 vs 0.048517708610531125 rel 0.0e+00 -> PASS
   [L partA:31] REF sod_mu_max: 0.16545799228159686 vs 0.16545799228159686 rel 0.0e+00 -> PASS
   [L partA:32] REF sod_cond: 94.92214093489962 vs 94.92214093489969 rel 7.5e-16 -> PASS
   ```
2. **Geometric-mean identity:** holds for every design, with largest relative deviation 3.5e-15 [L partA:33].
3. **Part B:**
   - M_min = 24 is bitwise Round 14's R(37, 13):
     ```
     [L partB:43] CHECK3 M_min = 24 vs Round 14 R(37, 13): {'n_j': True, 'rec24_bitwise': True, 'rec120_bitwise': True, 'pass_': True} -> PASS
     ```
   - Every fitted rule has E[N] ≤ B on σ/24 and on σ/120, and j + 1 is infeasible.
   - Largest mass deviation 8.7e-13 (σ/24) and 7.1e-13 (σ/120); min m = 0 [J checks.3_partB].
4. **Round 10:** f = 209.0685398686887 and u_cert = 209.07220154478713 are bitwise equal to the archive [L partA:25].

`git status` on rounds 4–14 and the frozen files was clean after every stage, and no empty folder was created under `round4/` [L partA:35], [L partB:72], [L summary:32].

## 6. Where the prompt and the code differ

I found no contradiction. Two notes:
- The rule's μ and cond from the outside continuous-rule Monte Carlo (0.262–0.290, ≈ 99.2) are close to the σ/24 lattice values here.
- The fixed threshold and class W have no evaluator forward. Their J comes from round 10's model (`ub.Problem.J`), as §1 prescribes, and their E[N] = 1 + Σ q and S = 1 + Σ c_max(q) are model values.

## 7. DECISION list

- **DECISION 17-1.** Scoring conventions, fixed in `predictions_round17.md` before running:
  - μ, cond and SD ratios are taken on the σ/24 J (round 10's model J for the fixed threshold and class W);
  - "about 3.1" means it rounds to 3.1, i.e. lies in [3.05, 3.15);
  - "monotone" means strictly increasing at every step 48 → 6, checked separately on σ/24, on σ/120 and for the Monte Carlo.
- **DECISION 17-2.** Constant SoD: J is the archived `round4/results/J_U_SoD.npy`. It was recomputed with the accepted operator and is bitwise equal; E[N] and S come from that operator.
- **DECISION 17-3.** The round-4 table is re-evaluated as Round 11's `inputs` stage did (`from_round4` with the old action set), with `eval13` (TOL24 1e-11). The tolerance only decides whether a run stops, so the value is unaffected.
- **DECISION 17-4.** T′ is decoded with round 11b's own `sprime.TableG` and `sprime.set_union()`, imported read-only. Their edge nodes LN/UN are equal to the arrays stored in the archived npz. It is evaluated with `eval13.evaluate`.
- **DECISION 17-5.** The fixed threshold and class W use J = `ub.Problem.J(q)` (round 10), with E[N] = 1 + Σ q and S = 1 + Σ c_max(q). Class W uses q̂ from `ub.solve(prob, B − 1, tol = 0.01)`.
- **DECISION 17-6.** Diagnostics:
  - μ from `scipy.linalg.eigh(J, J_full)`;
  - cond from `numpy.linalg.cond` (2-norm);
  - J⁻¹ by Cholesky;
  - parameter order θ = [G(0), Re G(jω₁), Im G(jω₁), …, Im G(jω₄)].
- **DECISION 17-7.** Part B:
  - the rule is Round 14's rule with the minimum half-width as a parameter, and `fit_rule17` is a copy of Round 14's `fit_rule` with that rule;
  - the Monte Carlo is Round 14's continuous routine with r-side distance m, R = 10⁴ in 20 batches with seeds 20261109 + b, the same for every m;
  - the scan starts at the lattice e_eff and steps by 0.04 σ towards E[N] = B, then E[N] and log det J are interpolated linearly between the bracketing pair.
- **DECISION 17-8.** Q2 "larger / equal / smaller" uses the quotient (μ_min ratio)/(geometric-mean ratio): above 1 means larger in the worst direction.
- **DECISION 17-9.** Q3 is answered for the designed triggers (round-4 table, T₁, T′, rule); constant SoD, the fixed threshold and class W are reported alongside.
- **DECISION 17-10.** Part B ran in the background, and its results were committed and pushed when it finished. The duplicate prompt you sent mid-run was byte-identical to the committed one, so the run continued without restarting.
- **DECISION 17-11.** Rounds 4–14 (the rounds present on this branch) were not modified. The following were imported read-only: round 13's `eval13`, round 14's `rule14`, round 11b's `sprime` (which itself imports round 11's `eval11` and `dp11`), and round 10's `ub`. Numba compiled-code caches under `__pycache__` are git-ignored (`evaluator/.gitignore`).

## 8. Wall times

- Part A: 75 s [L partA:34]. This covers four σ/24 forwards of about 20 s, one accepted-operator forward and round 10's solve.
- Part B: 1320 s [L partB:71]. Per M_min that is 4–5 σ/24 forwards of 20–22.5 s, one σ/120 forward of 119–133 s, and two Monte Carlo runs of 2.6–3.3 s.
- Summary: under 1 s [L summary:31].
- About 23 min in total.

## 9. Not done

- Part C (on hold): no DP and no certificate for a class with a smaller minimum distance.
- One input (U) and one budget (B = 316.892).
- No σ/120 diagnostics in Part A: they are on σ/24, as prescribed. Part B gives the σ/120 log det J of every rule.
