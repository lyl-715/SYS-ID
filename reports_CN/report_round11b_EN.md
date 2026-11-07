---
title: "Round 11b (addendum) report: the SoD-class bound on the larger class S ∪ S′"
date: "2026-11-07"
---

English copy of `report_round11b_CN.md`. Where the two differ, the Chinese version is the report of record.

**Task.** The Round 11b addendum prompt (given in the conversation; not committed as a file). Branch `round11-edge-actions`, continued from `395a11a`:
- predictions `round11/addendum/predictions_round11b.md` committed in `4c67843`, before any computation, with the scoring rules and two "problems found before running";
- code, check 1 and inputs in `aa60e5d`; scan and post results in `bf2c628`; first version of this report in `fd8e961`;
- the follow-up (Monte Carlo and shape statistics of T′, Proposition 5 at B + 0.25; section 12) in the commits after that.

Rounds 4–10 and round 11's `dp11.py`, `eval11.py` (and the other round-11 files) are read-only; all new code is in `round11/addendum/`. `git status` on those directories is clean after every stage [L 17, 28, 48, 95, 142].

**Scope.**
- Nominal truth θ₀ (θ* in the proposition), σ/24 discrete model, B = 316.8921168493721, p = 9.
- Tangent point A₁ = J(T₁)⁻¹, with T₁ = `round11/results/table_edge_iter.npz` (round 11's edge_j1). γ_k is bit for bit that of round 11's edge round-2 scan.
- **The class actually run is S ∪ S′** (per-node union, 884 actions), not the literal S′. The literal S′ does not contain S; see section 2 and DECISION 11b-2. You chose this change after the predictions commit and before any run.

**Sources.** Paths are relative to `code_8_3_stage2/stage2_pkg/evaluator/round11/addendum/`.
- `[L n]` is line n of `logs/run_round11b.log`. The post stage ran twice (DECISION 11b-10); this report cites the second run, lines 99–139. The follow-up mc stage also ran twice (DECISION 11b-14); the second run, lines 164–188, is cited.
- `[J a.b]` is field `a.b` of `results/round11b.json`.
- Per-λ records: `results/scan_union.json` and `results/archive_union.jsonl`.
- Code: `sprime.py` (action blocks, generic DP kernel, resumable scan), `mc11b.py` (Monte Carlo for the union action set) and `run_round11b.py` (driver; stages inputs, regress, scan, post, mc).

## 1. One-sentence conclusion

Over the class S ∪ S′, which contains round 11's S and the literal S′, the certified bound at the nominal budget is UB_SoD = **207.7097**. So **T₁ is within 0.3545 nat of the optimum of the class (D-efficiency ≥ 96.13 %)**, and within 0.3594 nat (≥ 96.08 %) at B + 0.25, the budget of Proposition 5's final statement (section 12.3).

The scan's primal table **T′ has I = 207.6467, 0.291 nat above T₁**, with σ/120 E[N] = 316.633 ≤ B. It is within **0.0631 nat of the bound (D-efficiency ≥ 99.30 %)** (0.0680 nat, ≥ 99.24 %, at B + 0.25) and is reported as the new best table. Its Monte Carlo passes against σ/120 (|z| ≤ 0.65, section 12.1).

Sources: [L 111–112, 118]; [J ub.UB, ub.gap, ub.deff_lower, ub.UB_Bprime, Tprime.\*].

## 2. The class S′ and S ∪ S′

**Literal S′ (DECISION 11b-1, fixed by the prompt).**
- Edge set E′ = {0.5, 0.75} ∪ {1.0, 1.1, …, 3.0} ∪ {3.5, 4, 5, 6, ∞}, 28 values.
- Minimum distance from r of 1σ (24 grid units): lower edge = min(x̄_k − e⁻σ, r − 1σ), upper edge = max(x̄_k + e⁺σ, r + 1σ), rounded outward to grid nodes.
- 784 actions.

**The literal S′ does not contain S.** The prompt says "E ⊂ E′, so the bound over S′ is ≥ the bound over S by construction". The inclusion fails, for two reasons:
1. The E values 1.25, 1.75 and 2.25 are not in E′, whose 0.1σ steps give 1.2/1.3 and so on. On the σ/24 grid 1.25σ is 30 grid units; neither 1.2σ = 28.8 nor 1.3σ = 31.2 reaches that node after outward rounding.
2. S's r-side edge r ∓ 2σ, which is the active edge on the far side almost everywhere, is not an S′ edge.

Measured: **97.7 % of T₁'s mass** at k ≥ 2 (97.6 % of its information) is on windows the literal S′ cannot represent. The lower edge is not representable for 65.5 % of the mass and the upper edge for 68.7 % [L 15; J inputs.T1_vs_Sprime].

**What was run: S ∪ S′ (DECISION 11b-2, your choice).**
- At every node the table may pick any of the 100 S actions (E, 2σ) or the 784 S′ actions (E′, 1σ): 884 actions.
- The S block comes first, so codes 0–99 are exactly round 11's codes and T₁ is a table of this class as it stands. The three row constructions agree at every (k, i) of the full grid, with 0 differences [L 14].
- Hence S ⊂ S ∪ S′ and S′ ⊂ S ∪ S′ by construction. The bound below holds for S′ and for S, and check D is a genuine consistency check.
- Cost: about 315–350 s per DP per λ (S: 42 s).

Edge-node checks [L 11–13]:
- outward rounding in physical units has 0 violations for S, S′ and the union;
- for non-integer 24e, u_k ∓ 24e is at least 3.5e-5 from the nearest integer, far above the 1e-9 criterion.

## 3. A: dual bound over S ∪ S′

Tangent point A₁ = J(T₁)⁻¹:
- I(T₁) = 207.35521730257756 and E[N] = 316.7261244018059, recomputed with eval11, are bit for bit round 11's [L 9];
- γ̄ = 0.006591104597771138 and γ₁ = 0.009447863192807387 are bit for bit those of `scan_edge_j2.json` [L 10].

The price scan uses the frozen logic: λ₀ = 5γ̄, doubling, ×0.8, 8 bisections. D(λ) = γ₁ + V_λ + λ(B − 1) and UB(λ) = I(T₁) − p + D(λ).

Per-λ table. Sources: [L 99–110]; [J per_lambda].

| run | phase | λ/γ̄ | λ | E[N] (σ/24) | feasible | V_λ | D(λ) | D(λ) + 0.1λ | log det | UB(λ) |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | start | 5.0 | 0.0329555 | 81.342 | yes | 0.896011 | 11.315849 | 11.319144 | 198.91481 | 209.67107 |
| 2 | scan | 4.0 | 0.0263644 | 154.257 | yes | 1.639458 | 9.977218 | 9.979855 | 203.16290 | 208.33244 |
| 3 | scan | 3.2 | 0.0210915 | 268.325 | yes | 2.720531 | 9.392629 | 9.394738 | 206.64257 | 207.74785 |
| 4 | scan | 2.56 | 0.0168732 | 428.861 | no | 4.155080 | 9.494647 | 9.496334 | 209.41769 | 207.84986 |
| 5 | bisect | 2.88 | 0.0189824 | 335.754 | no | 3.354094 | 9.359926 | 9.361825 | 207.99121 | 207.71514 |
| 6 | bisect | 3.04 | 0.0200370 | 302.045 | yes | 3.018749 | 9.357714 | 9.359717 | 207.35966 | 207.71293 |
| 7 | bisect | 2.96 | 0.0195097 | 318.849 | no | 3.182143 | 9.354541 | 9.356492 | 207.68454 | 207.70976 |
| 8 | bisect | 3.00 | 0.0197733 | 311.226 | yes | 3.099344 | 9.355025 | 9.357003 | 207.53991 | 207.71024 |
| 9 | bisect | 2.98 | 0.0196415 | 315.066 | yes | 3.140495 | 9.354535 | 9.356499 | 207.61306 | 207.70975 |
| **10** | bisect | **2.97** | **0.0195756** | **316.843** | yes | 3.161258 | **9.354477** | **9.356435** | **207.64665** | **207.70969** |
| 11 | bisect | 2.965 | 0.0195426 | 317.850 | no | 3.171683 | 9.354492 | 9.356447 | 207.66578 | 207.70971 |
| 12 | bisect | 2.9675 | 0.0195591 | 317.337 | no | 3.166466 | 9.354481 | 9.356437 | 207.65605 | 207.70970 |

Full-precision λ values are in [J per_lambda.\*.lam].

**Result.**
- **UB_SoD(S ∪ S′; B) = 207.709694**, attained at run 10 (λ = 0.01957558065538028 = 2.97γ̄, a bisection step).
- **gap = UB − I(T₁) = 0.354477 nat, e^{−gap/9} = 96.14 %** [L 111; J ub].
- Against round 11's UB_SoD(S) = 207.414762, the union bound is 0.2949 nat higher [L 116; J ub.UB_minus_UB_S].

## 4. B: budget convention B′ = B + 0.1

**Formula.** No new DP is needed:

UB(B′) = I(T₁) − p + min_λ [ γ₁ + V_λ + λ(B − 1) + 0.1 λ ]

i.e. add 0.1λ to each D(λ); V_λ does not depend on B.

| budget | UB | at | added vs B | gap(T₁) | e^{−gap/9} | gap(T′) | e^{−gap/9} |
|---|---:|---|---:|---:|---:|---:|---:|
| B | 207.709694 | run 10 | — | 0.354477 | 96.14 % | 0.063044 | 99.30 % |
| **B′ = B + 0.1** | **207.711652** | run 10 | +0.001958 | **0.356435** | **96.12 %** | **0.065002** | **99.28 %** |
| B + 0.25 | 207.714588 | run 12 | +0.004893 | 0.359371 | 96.09 % | 0.067938 | 99.25 % |
| B + 0.5 | 207.719478 | run 12 | +0.009783 | 0.364260 | 96.03 % | 0.072827 | 99.19 % |

Sources: [L 111–115]; [J ub.UB_Bprime, ub.UB_B_plus_delta]. The gaps are these UB values minus I(T₁) and I(T′).

**The premise does not hold (DECISION 11b-6).** The prompt states that σ/24 and σ/120 E[N] differ by < 0.1 for every table seen so far, so that B′ covers every table with σ/120 E[N] ≤ B. That is true only for old-action-set tables:
- `round7/results/partC_regrade.json` gives max_d1 = 0.078, and round 11's old tables have d1 = 0.072.
- For edge-action tables, d1 = E[N]₂₄ − E[N]₁₂₀ is 0.197 (edge_j0), 0.210 (edge_j1), 0.206 (edge_j2), and 0.2103 for this round's T′ [L 134; J budget_convention].

So B′ = B + 0.1 does **not** cover every edge-class table with σ/120 E[N] ≤ B. Covering the tables seen so far needs δ ≥ 0.211, e.g. δ = 0.25 (gap(T₁) 0.359, D-efficiency ≥ 96.09 %).

Proposition 5 itself is stated for σ/24 E[N] ≤ B′ and stays correct; only the reading "it covers every σ/120-feasible table" is affected. After the first version of this report you decided that Proposition 5's final statement uses **B + 0.25** (DECISION 11b-6); the corollary is restated accordingly (sections 9 and 12.3).

## 5. C: the scan's primal table T′

T′ is the scan's returned table: the largest F among the σ/24-feasible runs, i.e. run 10. Sources: [L 118, 117, 119]; [J Tprime, check2, packed].

| quantity | value |
|---|---:|
| λ (run 10) | 0.01957558065538028 (2.97γ̄) |
| E[N] σ/24 / σ/120 | 316.84334 / 316.63305 (≤ B, slack 0.259) |
| I σ/24 / σ/120 | 207.646650 / 207.644533 |
| I(T′) − I(T₁) (σ/24) | **+0.29143** |
| I₁₂₀(T′) − I₁₂₀(T₁) | +0.29145 |
| gap = UB − I(T′) (B / B′) | 0.06304 / 0.06500 |
| D-efficiency lower bound (B / B′) | 99.30 % / 99.28 % |
| mass share on S′-block actions | 88.5 % |
| of which: at least one edge < 2σ from r | 88.5 % |

- Since 0.291 > 0.05, T′ is reported as the new best table (prompt §3C) and packed as `results/table_edge_Sprime.npz` (995 kB, round trip equal).
- The bound of section 3 does not depend on which table the gap is taken against. A was not iterated again.
- I₁₂₀(T₁) = 207.35308 is from `round11/results/round11.json["tables"]["edge_j1"]["I120"]`.
- Descriptive: 88.5 % of T′'s mass uses S′-block actions, and each of those windows has at least one edge closer than 2σ to r. This says what T′ uses; it is not an attribution.

## 6. D: monotonicity of the minimum dual

**The prompt's criterion.** min D(S ∪ S′) = 9.354477 ≥ min D(S) − 1e-9 = 9.059544 − 1e-9; the difference is +0.2949. **Pass** [L 120].

The two λ grids differ, so strictly this comparison does not follow from S ⊂ S ∪ S′. At an identical λ the inclusion does imply V(S ∪ S′) ≥ V(S), so two pointwise comparisons were added:
- the first five λ are bit for bit those of round 11's S scan (5, 4, 3.2, 2.56, 2.88 γ̄), with V(S ∪ S′) − V(S) = +0.0427, +0.1029, +0.2116, +0.3842, +0.2910 [L 121–125];
- one extra S-block DP at the union's argmin λ (2.97γ̄, 37 s) gives D(S ∪ S′) − D(S) = +0.2645 [L 126].

All are positive [J checks.4_min_dual_monotone].

## 7. Predictions

The scoring rules are in `predictions_round11b.md` (`4c67843`). The predictions were written for S′; the class actually run is S ∪ S′ (section 2), and they are scored on it. Sources: [L 135–139]; [J predictions].

| # | prediction | value | result |
|---|---|---|---|
| 1 | UB_SoD − I(T₁) ∈ [0.06, 0.40] | 0.3545 | **hit** |
| 2 | the B′ correction adds < 0.005 nat | 0.00196 | **hit** |
| 3 | T′ improves on I(T₁) by < 0.20 nat | +0.2914 | **miss** |
| 3b | (consequence) certificate gap, whichever table, < 0.40 | max(0.3545, 0.0630) = 0.3545 | **hit** |
| 4 | argmin λ at or next to the final bisection step | run 10 of 12 | **miss** |

Line 4: the argmin falls at run 10, two steps before the last. D(λ) of runs 9–12 agree to within 5.8e-5 nat, i.e. the dual is flat there.

## 8. Checks 1–7

All pass [L 127–133, 184; J checks, all_checks_pass]. Check 7 is the follow-up Monte Carlo (section 12.1).

| # | check | result | source |
|---|---|---|---|
| 1 | regression: the S-block, 2σ code path reproduces round 11's edge round-2 scan | first two λ only (5γ̄, 4γ̄), as the prompt allows: run, phase, λ, E[N], N, F, V*_λ, S, log det, dual and feasibility all **bit for bit** | [L 24–26]; `results/state_regress.json` |
| 2 | forward = fixed-policy backward recursion for T′ (≤ 1e-12) | F 2.4e-13, N 2.5e-13, F − λN 2.2e-13 (relative); diagnostic: DP value − forward total = 1.2e-12 (absolute) | [L 117] |
| 3 | conservation, non-negativity, zero boundary mass on every forward | σ/24 max\|Σm − 1\| = 5.8e-13 (12 scan, 2 regression, T′), σ/120 1.4e-12 (≤ 1e-10), min m = 0; no assertion fired | [J checks.3_conservation] |
| 4 | minimum-dual monotonicity (D) | +0.2949; all pointwise supplements positive (section 6) | [L 120–126] |
| 5 | UB ≥ I(T₁) and ≥ I(T′) | 207.7097 ≥ 207.3552 and 207.6467 | [J checks.5_UB_ge_I] |
| 6 | σ/120 E[N] of T′ ≤ B | 316.633 ≤ 316.892 | [J checks.6_budget120_Tprime] |
| 7 (follow-up) | Monte Carlo on T′ (DECISION 11-7) | 10⁴ trajectories; z against σ/120: E[N] −0.01, S +0.59, log det +0.65 (\|z\| ≤ 3); 0 clipped lookups; the same code reproduces round 11's MC on T₁ bit for bit | [L 182–184]; [J mc_Tprime, mc_T1_regression] |
| extra | outward rounding of edge nodes | 0 violations for all three sets | [L 11–13] |

The status string "run cap (40 DP runs) reached" in check 1 is dp11's hard-coded text; the cap in this run was 2.

## 9. Proposition 5 (ready to paste)

The Chinese text is in the CN report, section 9.

> **Proposition 5.** Let S ∪ S′ be the class of triggers whose window at step k ≥ 2 and reference value r_k is an interval containing r_k of one of two kinds:
> (i) lower edge min(x̄_k − e⁻σ, r_k − σ) and upper edge max(x̄_k + e⁺σ, r_k + σ) with e± ∈ E′;
> (ii) the same form with e± ∈ E and distance 2σ from r_k.
>
> Both edges are rounded outward to the σ/24 grid, and the window may be any function of the receiver-known history. For every T ∈ S ∪ S′ with E[N] ≤ B′ on the σ/24 discrete model at θ*, and every λ ≥ 0,
>
> log det J(T) ≤ log det J(T₁) − p + γ₁ + V_λ + λ(B′ − 1),
>
> where γ_k = σ⁻² h_kᵀ J(T₁)⁻¹ h_k and V_λ is the value of the Bellman recursion (15) over the actions of S ∪ S′ at price λ, started from g₁. In particular the bound holds for every T ∈ S′ and every T ∈ S.
>
> **Proof.** Lemma 3 at J(T₁); eq. (12); Lemma 1; and the exactness of finite-horizon dynamic programming for additive expected rewards over Markov policies, which suffice (Bertsekas, Dynamic Programming and Optimal Control, Vol. I, Ch. 1). V_λ is evaluated in floating point; the forward–backward agreement bounds its rounding at 1e-12 relative.

**Numerical corollary (final statement, B′ = B + 0.25; DECISION 11b-6).** At the nominal budget, with B′ = B + 0.25 (which covers the σ/120 budget of every edge-class table seen so far, section 4):
- T₁ is within **0.3594 nat** of the optimum of S ∪ S′ (D-efficiency ≥ **96.08 %**);
- T′ is within **0.0680 nat** (D-efficiency ≥ **99.24 %**).

For reference, with the prompt's original B′ = B + 0.1: T₁ 0.3565 nat (≥ 96.11 %), T′ 0.0651 nat (≥ 99.28 %). Gaps are rounded up and D-efficiencies down (DECISION 11b-16). Sources: [L 185]; [J proposition5, ub.UB_B_plus_delta].

## 10. Decisions

- **DECISION 11b-1 (fixed by the prompt, unchanged).** E′ with 28 values, 1σ minimum distance from r, 784 actions.
- **DECISION 11b-2 (class; your choice).** Before running, the literal S′ was found not to contain S (section 2). Three options were put to you and you chose "S ∪ S′": the per-node union, 884 actions, S block first. Since S′ ⊂ S ∪ S′ the bound also holds for the literal S′, but it may be slightly looser than a bound computed over S′ alone. A scan over S′ alone was not run.
- **DECISION 11b-3 (edge nodes).** LN_k(e) = ⌊u_k − 24e⌋ and UN_k(e) = ⌈u_k + 24e⌉, with 24e formed as an exact rational and converted to the nearest double.
  - For integer 24e these equal round 11's fl_k − 24e and cl_k + 24e bit for bit (asserted).
  - For non-integer 24e, u_k ∓ 24e is asserted to be > 1e-9 from an integer (smallest observed 3.5e-5).
  - Windows: L = max(i − LN_k(e⁻), MIN), U = max(UN_k(e⁺) − i, MIN); e = ∞ gives MIN.
- **DECISION 11b-4 (codes and ties).** Code = off_b + (j_lo − lo_b)·n_b + (j_hi − lo_b), with the S block first (codes 0–99 = round 11's). Ties go to the first code (strict >), as in dp11.
- **DECISION 11b-5 (reuse).**
  - `DesignG` subclasses `dp11.Design11` and adds only `dp_g`.
  - `_dp_step_u` is a copy of `dp11._dp_step` with three changes: half-widths from per-step edge nodes, a clamp at the block's MIN, and a loop over blocks. The floating-point expressions and their order are unchanged.
  - `ScanG` subclasses `dp11.Scan11` and overrides only `run()`: generic DP and table, a JSONL archive per λ (fsync) for resuming, and only the current best feasible table kept on disk. `scan()`'s λ logic is reused unchanged.
  - Forward, backward and σ/120 use `eval11` directly (they accept any `rows_full`).
- **DECISION 11b-6 (budget convention).** B′ = B + 0.1 is computed as the prompt asks. UB(B + δ) is also reported for δ = 0.25 and 0.5, and the report notes that the prompt's < 0.1 premise fails for edge-class tables (section 4). After the first version of this report you decided that Proposition 5's final statement uses **B + 0.25**; the corollary is restated accordingly (sections 9 and 12.3), with the B + 0.1 numbers kept for reference. Your message labelled this "DECISION 11b-3"; 11b-3 is already the edge-node rule in this report and in `sprime.py`, so it is recorded here to avoid a clash.
- **DECISION 11b-7 (check D supplements).** Besides the prompt's comparison of minima: pointwise comparisons at the shared λ, and one extra S-block DP at the union's argmin λ (37 s).
- **DECISION 11b-8 (T′).** T′ is the scan's final table (largest F among σ/24-feasible runs, ties to the smaller λ), evaluated on σ/24 and σ/120. Since +0.291 > 0.05 it is reported as the new best table and packed. A was not iterated again.
- **DECISION 11b-9 (γ).** `scan_edge_j2.json` stores only γ̄ and γ₁, not the γ vector. So J(T₁) was computed with eval11 on σ/24, γ was obtained with `Design11.gamma_from_J`, and γ̄ and γ₁ were checked bit for bit. The γ vector is saved as `results/gamma_A1.npy`.
- **DECISION 11b-10 (job handling).**
  - The first background launch started in the wrong working directory and never ran the script; it was relaunched with an absolute path.
  - The scan ran its 12 λ in one go, so resuming was never needed: 4236 s, each DP 315–350 s.
  - The post stage ran twice. The second run only changed how check 3 is recorded (it now includes the regression forwards and no longer uses a hand-coded forward count); every other number is identical. Log lines 52–92 are the first run and 99–139 the second.
- **DECISION 11b-11 (T₁ representability in the literal S′).** At node i the lower edge is representable iff L = 24 or i − L ∈ {LN_k(e) : e ∈ E′ finite}; the upper edge likewise. Shares are weighted by the forward mass at k ≥ 2, with the information m·c as a supplement.
- **DECISION 11b-12 (prediction scoring).** The predictions were written for the literal S′ and are scored on S ∪ S′, the class actually run (section 7).
- **DECISION 11b-13 (Monte Carlo implementation).**
  - `mc11b._mc_batch_u` is a copy of `eval11._mc_batch` with one change: the half-widths of the looked-up node come from the union action set (per-step edge nodes LN, UN and the block's MIN).
  - Everything else is round 11's: nearest node (ties to the smaller index), window centred at the continuous r, cfun, random-number order, seeds 20261109 + b, 20 batches × 500, 4 workers.
  - Validation: rerun on T₁, it reproduces `round11.json` checks.4_monte_carlo bit for bit (EN_mc, EN_se, S_mc, S_se, logdet_mc, logdet_batch_se, n_clipped).
  - The seeds are round 11's, so the T₁ and T′ runs use common random numbers. The pass rule is unchanged: |z| ≤ 3 against σ/120 and no clipped lookups.
- **DECISION 11b-14 (the mc stage ran twice).** The first run (log lines 143–163) stopped before writing anything, on an internal consistency assertion: UB(B + 0.25), recomputed in a different summation order, differed from post's value in the last bit, and the assertion required exact equality. After switching to post's order with a 1e-12 tolerance, the second run (lines 164–188) wrote the results. The MC and shape numbers are identical in both runs.
- **DECISION 11b-15 (shape statistics).** `eval11.Stats` with γ weights at A₀ = J_SoD⁻¹, as in `round11/post11.analyse`. T₁'s statistics through the union action set equal `round11.json` tables.edge_j1 bit for bit (quantiles, all bin fields, support width, I) [L 181].
- **DECISION 11b-16 (rounding of bounds).** In prose bounds of the form "within X nat, D-efficiency ≥ Y %", gaps are rounded up to 4 decimals and D-efficiencies down to 0.01 %; tables keep 6 decimals. The first version of this report rounded to nearest in sections 1 and 9 (e.g. 0.354477 written as 0.354); this is corrected in sections 1, 9 and 12.

## 11. Not done in this round

- A scan over the literal S′ alone (784 actions), and a primal table within S′.
- Another iteration of A (not asked for), and a bound with J(T′) as the tangent point.
- Mismatch sweep and margin design for T′ (Round 12 material). T′'s Monte Carlo and shape statistics are in section 12.
- Bounds away from the nominal truth.

**Outputs.** Under `round11/addendum/` (the source paths are relative to it):
- `results/round11b.json`: per-λ table, bounds for B and B′ (and B + δ), gaps, D-efficiency, T′ numbers, checks 1–7, predictions, budget-convention evidence; follow-up fields mc_Tprime, mc_T1_regression, shape, proposition5, and D_B025, UB_B025 in per_lambda;
- `results/table_edge_Sprime.npz`;
- `results/scan_union.json`, `results/archive_union.jsonl`, `results/state_inputs.json`, `results/state_regress.json`, `results/gamma_A1.npy`;
- `logs/run_round11b.log`.

## 12. Follow-up: Monte Carlo and shape statistics of T′, and the budget of Proposition 5

This section was added after the first version of the report. Sources: log lines 164–188 (second run of the mc stage) and the follow-up fields of `results/round11b.json`.

### 12.1 Monte Carlo (DECISION 11-7)

Same procedure as round 11 (DECISION 11b-13):
- 10⁴ trajectories, 20 batches × 500, seeds 20261109 + b, 4 workers;
- runtime lookup rule: nearest node, ties to the smaller index, window centred at the continuous r;
- pass rule: |z| ≤ 3 against σ/120 and no clipped lookups.

Regression first: the same code, run on T₁ (looked up through the union action set), reproduces `round11.json` checks.4_monte_carlo bit for bit [L 182; J mc_T1_regression].

| quantity | MC (± s.e.) | σ/120 | z (σ/120) | σ/24 | z (σ/24) |
|---|---:|---:|---:|---:|---:|
| E[N] | 316.6347 ± 0.1716 | 316.6330 | **−0.01** | 316.8433 | +1.22 |
| S | 1384.0752 ± 0.0493 | 1384.1041 | **+0.59** | 1384.4325 | +7.25 |
| log det J | 207.64430 ± 0.00035 | 207.64453 | **+0.65** | 207.64665 | +6.62 |

0 clipped lookups. **Pass** [L 183–184; J mc_Tprime, checks.7_monte_carlo_Tprime].

The z-scores against σ/24 are large for S and log det (7.2 and 6.6), the same pattern as T₁ in round 11 (+14.0 and +12.9 then):
- the σ/24 quadrature is 0.36 above the continuous-r process in S and 0.0023 nat above in log det; σ/120 agrees with the MC;
- 0.0023 nat is small against T′'s 0.291 nat over T₁ and against its 0.068 nat gap to the bound.

### 12.2 Shape statistics (round 11's definitions)

Definitions:
- facing edge = near-side half-width − |r − x̄_k| (σ units; positive means x̄_k is inside the window), mass-weighted over (k ≥ 2, i);
- bins of |r − x̄_k|/σ;
- "far-side half-width" is the half-width on the side away from x̄_k, mass-weighted.

Statistics come from `eval11.Stats` (DECISION 11b-15). T₁'s numbers equal `round11.json` bit for bit [L 181].

| table | facing edge 10 / 50 / 90 % (σ) | mean | mass in ≥ 40σ bin | sends there (share) | info per send there (unweighted c) | facing edge there (mean) | mean support width |
|---|---|---:|---:|---:|---:|---:|---:|
| T₁ | 1.28 / 1.53 / 1.78 | 1.558 | 15.0 % | 50.4 (16.0 %) | 4.11 | +1.52 | 1116.3σ |
| T′ | 1.41 / 1.62 / 1.81 | 1.601 | 19.5 % | 48.8 (15.4 %) | 4.55 | +1.67 | 1116.3σ |

Sources: [L 167–180]; [J shape.T1, shape.Tprime].

By bin (T₁ / T′):

| \|r − x̄_k\|/σ | mass share | send share | info per send | far-side half-width (σ) | facing edge (σ) |
|---|---:|---:|---:|---:|---:|
| [0, 2) | 20.7 % / 22.8 % | 18.8 % / 27.0 % | 4.63 / 4.65 | 2.00 / 1.14 | 1.64 / 1.68 |
| [2, 5) | 15.9 % / 15.4 % | 15.5 % / 16.8 % | 4.25 / 4.09 | 2.00 / 1.03 | 1.56 / 1.50 |
| [5, 10) | 15.3 % / 13.4 % | 15.4 % / 13.8 % | 4.19 / 4.16 | 2.00 / 1.14 | 1.54 / 1.53 |
| [10, 20) | 17.1 % / 14.6 % | 17.6 % / 14.1 % | 4.16 / 4.26 | 2.00 / 1.23 | 1.53 / 1.56 |
| [20, 40) | 16.0 % / 14.3 % | 16.8 % / 12.8 % | 4.12 / 4.36 | 2.00 / 1.20 | 1.52 / 1.60 |
| [40, ∞) | 15.0 % / 19.5 % | 16.0 % / 15.4 % | 4.11 / 4.55 | 2.00 / 1.16 | 1.52 / 1.67 |

Descriptive reading, no attribution:
- **Facing edge.** T′'s facing edge sits further out: 10 % quantile 1.28 → 1.41σ, median 1.53 → 1.62σ, 90 % quantile 1.78 → 1.81σ, mean 1.558 → 1.601σ.
- **Far side.** T₁'s far-side half-width is exactly 2σ in every bin (S's r-side minimum). T′'s is 1.03–1.23σ (88.5 % of T′'s mass uses S′-block actions, whose r-side minimum is 1σ; section 5).
- **[0, 2) bin.** States whose reference value is within 2σ of the output carry 27.0 % of T′'s transmissions, against 18.8 % for T₁.
- **≥ 40σ bin.** T′ has more mass there (19.5 % against 15.0 %) and more information per send (4.55 against 4.11).
- Both tables have the same mean support width (1116.3σ): the mass support is set by the reach of the measurement kernel, not by the table.

### 12.3 Final statement of Proposition 5: B′ = B + 0.25

Per your decision (DECISION 11b-6), UB(B + 0.25) = I(T₁) − p + min_λ [γ₁ + V_λ + λ(B − 1) + 0.25 λ] = **207.714588**, attained at run 12 (λ = 0.019559102893885853, a bisection step). Sources: [L 185]; [J proposition5]; D(λ) + 0.25λ per λ in [J per_lambda.\*.D_B025].

| table | gap = UB(B + 0.25) − I | D-efficiency lower bound |
|---|---:|---:|
| T₁ | 0.359371 | 96.0857 % |
| T′ | 0.067938 | 99.2480 % |

B + 0.25 covers every edge-class table seen so far with σ/120 E[N] ≤ B: the largest E[N]₂₄ − E[N]₁₂₀ is T′'s 0.2103 (section 4).

Restated corollary, ready to paste (Proposition 5 itself is in section 9, with B′ = B + 0.25):

> **Corollary.** At the nominal budget B, with B′ = B + 0.25: T₁ is within 0.3594 nat of the optimum of S ∪ S′ (D-efficiency ≥ 96.08 %); T′ is within 0.0680 nat (D-efficiency ≥ 99.24 %).

The Chinese text of the corollary is in the CN report, section 12.3.
