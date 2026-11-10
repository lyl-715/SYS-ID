---
title: "Round 13, Part A report: does the designed trigger's information depend on the input shape? (five of six inputs)"
date: "2026-11-10"
---

English copy of `report_round13_partA_CN.md`. Where the two differ, the Chinese version is the report of record.

**Prompt.** `reports_CN/prompt_round13_partA_input_invariance_EN.md`, with Part 0's rules. Branch `round13-input-trigger`. Commits:
- `614e8f8`: `round13/predictions_partA.md`, committed before any Part A computation;
- `5743c1f`, `a57b5cd`: Part A code;
- `2aaf176`: U;
- `f4ce8c8`, `ec9a6d9`: SCH and P5514 stopped at the DP constructor (§6 item 1);
- `1f84805`: DECISION 13-22 (dp13 fix, your decision) with its neutrality check for U;
- `d21d1dc`, `e4d3237`, `f996bfa`, `8874238`: SCH, P5514, LO, HI;
- `0f0e55e`: N stopped (§6 item 2);
- `3f561a8`: summary;
- `0d19866` … `04215bd`: checkpoint commits while inputs were running;
- this report: the commit after that.

**Scope.**
- Six inputs (U, P5514, SCH, LO, HI, N), budget B = 316.8921168493721, edge action set (100 actions), 9×9 `log det J` criterion.
- I_full = log det(σ⁻² Σ_k h_k h_kᵀ). Every other I is the evaluator's log det J at θ₀.
- Derived quantities (c, gain, S, information per transmission) are on **σ/120**, with σ/24 alongside (DECISION 13-19).
- "Supported" in this report only means the inputs behaved as the hypothesis predicts. No causes are claimed.

**Sources.** Paths are relative to `code_8_3_stage2/stage2_pkg/evaluator/`.
- `[L X:n]` is line n of `round13/logs/partA_X.log`, and `[L S:n]` is line n of `round13/logs/partA_summary.log`.
- `[J X.a]` is field `a` of `round13/results/partA_X.json`, and `[J S.a]` is field `a` of `round13/results/partA_summary.json`.
- Code: `round13/run13.py` (stage `partA`, `dpfix`, `partA_summary`), `round13/common13.py`, `round13/partA13.py`, `round13/dp13.py`.
- Per-input scans and J matrices are in `round13/results/partA_X/`. The packed best tables are `round13/results/table_A_X.npz`.

## 1. Verdict

**Incomplete: N has no designed table.** Its first scan stopped on the σ/24 mass-conservation assertion: \|Σm − 1\| = 1.004 × 10⁻¹² > 1e-12 at step 4217 of run 4 (§6 item 2).

**On the five inputs that have a designed table** (U, P5514, SCH, LO, HI), H1, H2, H3 and the resolution condition all hold:
- c_tab = I_table − I_full lies in [−11.672, −11.637], a range of **0.035 nat**;
- S_table is within **0.22 %** of U's;
- every certificate gap is ≤ 0.210 nat.

Under the matched constant SoD, c_sod ranges over **0.976 nat** (all six inputs).
Sources: [L S:9–13]; [J S.verdict].

## 2. Cross-input table

All values are on σ/120 except the "σ/24" columns. M\* is the matched half-width in grid units (σ/24); "M\*/σ" is the threshold in σ. Sources: [L S:3–8]; [J S.rows, S.partial].

| input | I_full | SoD at Δ\*: E[N] σ/24 | SoD at Δ\*: I σ/24 | matched SoD: M\* (M\*/σ) | E[N] | I_SoD | c_sod |
|---|---:|---:|---:|---:|---:|---:|---:|
| U | 219.0085 | 316.89 | 197.2010 | 480 (20.00) | 316.891 | 197.2012 | −21.807 |
| P5514 | 219.0080 | 316.47 | 197.9022 | 479 (19.96) | 317.136 | 197.9226 | −21.085 |
| SCH | 219.0085 | 314.81 | 197.7985 | 477 (19.88) | 316.647 | 197.8515 | −21.157 |
| LO | 218.0343 | 367.81 | 197.3527 | 561 (23.38) | 317.117 | 195.9725 | −22.062 |
| HI | 218.0343 | 274.27 | 195.4602 | 414 (17.25) | 316.915 | 196.7916 | −21.243 |
| N | 218.4169 | 423.97 | 199.3149 | 651 (27.13) | 317.014 | 196.6621 | −21.755 |

| input | table T\* | E[N] | S | I_table | I_table σ/24 | c_tab | certificate gap (σ/24) | gain = I_table − I_SoD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| U | edge_j1 | 316.516 | 1343.89 | 207.3531 | 207.3552 | −11.655 | 0.210 | 10.152 |
| P5514 | edge_j1 | 316.594 | 1341.00 | 207.3363 | 207.3386 | −11.672 | 0.114 | 9.414 |
| SCH | edge_j1 | 316.551 | 1344.01 | 207.3662 | 207.3684 | −11.642 | 0.084 | 9.515 |
| LO | edge_j1 | 316.676 | 1342.91 | 206.3717 | 206.3739 | −11.663 | 0.204 | 10.399 |
| HI | edge_j1 | 316.218 | 1346.07 | 206.3975 | 206.3996 | −11.637 | 0.125 | 9.606 |
| N | — (stopped) | — | — | — | — | — | — | — |

Information per transmission, (S − 1)/(E[N] − 1) on σ/120 [J S.info_per_send]:

| | U | P5514 | SCH | LO | HI | N |
|---|---:|---:|---:|---:|---:|---:|
| designed table | 4.256 | 4.246 | 4.256 | 4.251 | 4.267 | — |
| matched SoD | 1.473 | 1.540 | 1.541 | 1.437 | 1.543 | 1.432 |

Design details:
- In every input T\* is the table of the second scan (j1). No third scan was needed; the largest gap was 0.210 nat.
- The certificate UB′ (B + 0.25) per input: U 207.5653, P5514 207.4525, SCH 207.4520, LO 206.5781, HI 206.5243 [J X.design.certificate].
- σ/24 summary values: range c_tab 0.0347 nat; max S ratio 0.0021 [J S.range_c_tab24, S.S_ratio24_max_abs].

## 3. H1–H3 (five inputs with a table, σ/120)

| | rule | value | holds |
|---|---|---|---|
| H1 | max_i abs(S_table(i)/S_table(U) − 1) ≤ 0.03 | 0.0022 (P5514: −0.22 %; LO −0.07 %, HI +0.16 %, SCH +0.01 %) | yes |
| H2 | range c_tab ≤ 0.5 nat | 0.035 nat (−11.672 P5514 … −11.637 HI) | yes |
| H3 | abs(I_table(SCH) − I_table(U)) ≤ 0.2; abs(I_table(P5514) − I_table(U)) ≤ 0.2; abs(c_tab(LO) − c_tab(HI)) ≤ 0.5 | +0.013; −0.017; −0.026 nat | yes |
| resolution | certificate gap ≤ 0.5 nat for every table | 0.084 – 0.210 nat | yes |

The §6 verdict needs all six inputs. With N missing it is **incomplete**. The "not supported" condition (range c_tab > 1.5 nat) does not hold on the five inputs.

Rank agreement (DECISION 13-21: values within 0.1 nat are tied; two orders agree iff every pair has the same tie-aware relation) [L S:10–11]:
- **Designed table (five inputs).** By I_full: U = SCH = P5514 > HI = LO. By I_table: SCH = U = P5514 > HI = LO. The orders **agree**.
- **Matched SoD (six inputs).** By I_full: U = SCH = P5514 > N > HI = LO. By I_SoD: P5514 = SCH > U > HI > N > LO. The orders **do not agree**: one discordant pair (HI–N) and three pairs tied in one order but strict in the other (U–P5514, U–SCH, LO–HI).

## 4. Constant-SoD contrast

Pairwise differences in nat, σ/120 [L S:12]:

| | designed table | matched SoD | I_full |
|---|---:|---:|---:|
| I(SCH) − I(U) | +0.013 | +0.650 | 0.000 |
| I(P5514) − I(U) | −0.017 | +0.721 | −0.0005 |
| c(LO) − c(HI) | −0.026 | −0.819 | (I_full(LO) − I_full(HI) = −2.8e-14) |

- Range of c_sod: 0.976 nat over all six inputs. Over the five inputs that have a table it is also 0.976 nat. For c_tab it is 0.035 nat.
- At Δ\* the constant SoD's E[N] ranges from 274 (HI) to 424 (N). To bring E[N] to B, the threshold goes from 17.25σ (HI) to 27.13σ (N).
- With the table, every E[N] on σ/120 lies in [316.22, 316.68].
- gain = I_table − I_SoD is 9.41–10.40 nat over the five inputs.

## 5. Predictions (`round13/predictions_partA.md`, committed in `614e8f8`)

**3 hits, 0 misses, 4 undetermined** [L S:14–21]. "Undetermined" means the line needs N's designed table and holds on the other five inputs (DECISION 13-28).

| line | value | result |
|---|---|---|
| H1 holds (S within ±3 %) | 0.0022 on five inputs | undetermined |
| H2 holds, with σ/24 points P5514/SCH 207.35 ± 0.3, LO/HI 206.38 ± 0.3, N 206.76 ± 0.3 | range 0.035; P5514 207.339, SCH 207.368, LO 206.374, HI 206.400 (all inside); N missing | undetermined |
| H3 holds | +0.013, −0.017, −0.026 | **hit** |
| constant SoD not separable: range c_sod ≥ 0.7 and abs(c_sod(LO) − c_sod(HI)) ≥ 0.5 | 0.976; 0.819 | **hit** |
| I_SoD(P5514) − I_SoD(U) ∈ [0.5, 0.9] | 0.721 | **hit** |
| certificate gap ≤ 0.3 for every input after two scans | 0.084–0.210 on five inputs | undetermined |
| gain ∈ [9, 11.5] for every input | 9.41–10.40 on five inputs | undetermined |

## 6. Where the prompt and the code at 1b88f01 differ

1. **DP constructor: constant kernel-window width.** `Design11.__init__`, copied unchanged from round 11, asserts that every step k ≥ 2 has an 88σ kernel window of the same width. The first run of SCH and P5514 stopped there [L SCH:15], [L P5514:16]. At the step where the output reaches its minimum (x_k = min x), (x_k ∓ 88σ − z₀)/h computes to 1728.000000000018 / 5951.999999999982 instead of exactly 1728 / 5952. Both end nodes therefore fall outside, and the window has 4223 nodes instead of 4224.
   - For U, LO and HI the minimum is at step 1. Step 1 is the forced send, and its window is not used by the DP, so these three inputs were not affected.
   - You chose the minimal fix, recorded as **DECISION 13-22**. Its records, checks and neutrality proof are in §6.1.
   - Part 0's gates had built `Design11` only for U and LO, so this was not caught before Part A. That was a gap in my Part 0 coverage.
2. **σ/24 conservation tolerance for N (open).** `TOL24 = 1e-12` (Round 11, DECISION 11-5) stopped N in scan j0. The σ/24 forward of run 4's table (λ = 2.56γ̄, E[N] = 366.1, infeasible) reached \|Σm − 1\| = 1.004e-12 at step 4217 [L N:29]. Per Part 0 §0 this stops the run, and the tolerance was not changed.
   - A diagnostic with the tolerance relaxed to 1e-10 for information only (`round13/diag_massdev.py`, `round13/logs/diag_massdev_N.log`) gives these per-table maxima:

     | N table (scan j0) | E[N] | max abs(Σm − 1) | steps over 1e-12 | min m |
     |---|---:|---:|---:|---:|
     | run 1 | 76 | 2.8e-13 | 0 | 0 |
     | run 2 | 140 | 2.9e-13 | 0 | 0 |
     | run 3 | 236 | 3.3e-13 | 0 | 0 |
     | run 4 | 366 | 1.020e-12 (step 4253) | 121 of 4999 | 0 |

   - For comparison, over the other five inputs the largest σ/24 deviation among the reported forwards is 6.5e-13 (U, T₀). Among every σ/24 forward, scan runs included, it is 9.1e-13 (P5514, scan j1 run 4: an infeasible table with E[N] = 391; `mass_dev_max` in the scan jsons). In both N and P5514 the largest deviations come from infeasible, high-E[N] tables.
   - Completing N needs your decision (§9).
3. **Memory** was handled by DECISION 13-17 (your decision). `Stats` ran for every T\*, with peaks of 6.0–8.7 GiB per process (§7).

### 6.1 DECISION 13-22 records (SCH, P5514, N)

For each affected input there is exactly one step with 4223 nodes, always [KA, KB] = [1729, 5951] [L SCH:23], [L P5514:24], [L N:24]:

| input | step k (output minimum) | lost end node (relative to the exact-arithmetic window [1728, 5952]) | padded entry | kernel at KA − 1 and KB + 1 | padded row = evaluator kernel | sequential sums | limit check outside the cache range |
|---|---:|---|---|---|---|---|---|
| SCH | 2849 | both ends (1728 and 5952) | node KB + 1 = 5952 (upper end) | 0.0 / 0.0 exactly | bitwise | equal (1.0) | passes (u = −128.00 / +128.04) |
| P5514 | 4676 | both ends | node 5952 (upper end) | 0.0 / 0.0 exactly | bitwise | equal (1.0) | passes |
| N | 2717 | both ends | node 5952 (upper end) | 0.0 / 0.0 exactly | bitwise | equal (1.0) | passes |

The extra cached node KB + XC + 1 has abs(u) = 128.0 > 127, so ndtr and φ are at their exact limits there; these are the values the frozen rule would use. Neutrality for U [`round13/logs/dpfix_U.log`; `round13/results/dpfix13_U.json`]:
- (a) Every `Design11` cache of dp13 (UC, PPC, PNC, PHC, GK, g1, W88, NC, a1, b1) is bitwise equal to dp11's.
- (b) One DP run at λ = 0.06022857891938863 (Round 11's final λ of scan j0) reproduces the `table_edge_A0.npz` codes (0 differing nodes) and dsum = 11.273474443508011 bitwise.

End-to-end check, as you required: forward score against fixed-policy backward score of every final table of SCH and P5514 (relative, rule ≤ 1e-12). N has none.
```
[L SCH:38]   edge_j0: EN120 316.0952081868754 (<= B: True); mass dev 3.30e-13 / 1.08e-12; forward-backward F 3.3e-13 N 3.4e-13 total 3.1e-13 (PASS)
[L SCH:53]   edge_j1: EN120 316.5512743314831 (<= B: True); mass dev 3.41e-13 / 9.55e-13; forward-backward F 3.4e-13 N 3.5e-13 total 3.3e-13 (PASS)
[L P5514:39] edge_j0: EN120 316.2377406383672 (<= B: True); mass dev 4.02e-13 / 1.02e-12; forward-backward F 2.0e-13 N 2.2e-13 total 1.8e-13 (PASS)
[L P5514:54] edge_j1: EN120 316.59370108041276 (<= B: True); mass dev 5.45e-13 / 8.27e-13; forward-backward F 3.4e-13 N 3.4e-13 total 3.3e-13 (PASS)
```

## 7. Checks (Part A §8)

1. **Forwards.** Every reported forward of the five inputs passed the asserted mass conservation, non-negativity and zero-mass-at-the-grid-ends checks. The largest σ/24 deviation was 6.5e-13 (U), and the largest σ/120 deviation 1.27e-12 (LO T₀; tolerance 1e-10). N stopped on this check (§6 item 2). Source: [J X.checks.1_forwards].
2. **Scans.** All 10 scans ended with status `done`, i.e. no monotonicity stop. Forward against fixed-policy backward for every final table: U, LO and HI are quoted below; SCH and P5514 are in §6.1. The worst value is 5.5e-13.
   ```
   [L U:19]  edge_j0: EN120 316.413413652473 (<= B: True); mass dev 6.55e-13 / 8.72e-13; forward-backward F 5.2e-13 N 5.2e-13 total 5.1e-13 (PASS)
   [L U:21]  edge_j1: EN120 316.5159931346839 (<= B: True); mass dev 5.12e-13 / 7.89e-13; forward-backward F 4.3e-13 N 4.3e-13 total 4.1e-13 (PASS)
   [L LO:41] edge_j0: EN120 316.4483126782755 (<= B: True); mass dev 2.72e-13 / 1.27e-12; forward-backward F 3.3e-13 N 3.4e-13 total 3.1e-13 (PASS)
   [L LO:56] edge_j1: EN120 316.6761308617272 (<= B: True); mass dev 4.01e-13 / 1.01e-12; forward-backward F 3.5e-13 N 3.6e-13 total 3.2e-13 (PASS)
   [L HI:34] edge_j0: EN120 316.51063102372206 (<= B: True); mass dev 5.95e-13 / 7.05e-13; forward-backward F 5.4e-13 N 5.3e-13 total 5.5e-13 (PASS)
   [L HI:49] edge_j1: EN120 316.2175684345478 (<= B: True); mass dev 3.75e-13 / 9.52e-13; forward-backward F 3.8e-13 N 3.9e-13 total 3.7e-13 (PASS)
   ```
3. **Reported tables.** E[N] on σ/120 is ≤ B for every T\* (316.218–316.676). I(T) ≤ UB′ holds for T₀ and T₁ of every input [J X.checks.3_table].
4. **Matched SoD** was accepted for all six inputs. Residuals E[N]₁₂₀ − B: U −0.001, P5514 +0.244, SCH −0.245, LO +0.225, HI +0.023, N +0.122 [L X: MATCHED line].
5. **U reproduces Round 11:** I_table σ/24 = 207.35521730257756 (equal) [J U.checks.5_U_round11]. With DECISION 13-22 the U path is bitwise unchanged (§6.1).
6. **abs(I_full(LO) − I_full(HI))** = 2.8e-14 ≤ 0.01 [J S.check6_LO_HI_I_full].

Shape statistics of T\* (`eval13.Stats`, γ of the j0 tangent, computed after the design was released; DECISION 13-17/13-18) [L X: STATS line]:

| input | facing-edge quantiles 10 / 50 / 90 % (σ) | share of transmissions in the abs(r − x̄) ≥ 40σ bin | node values | process peak RSS |
|---|---|---:|---:|---:|
| U | 1.280 / 1.526 / 1.778 | 16.0 % | 1.34 × 10⁸ | 7.6 GiB |
| P5514 | 1.281 / 1.527 / 1.777 | 14.7 % | 1.23 × 10⁸ | 7.0 GiB |
| SCH | 1.282 / 1.526 / 1.777 | 14.5 % | 1.30 × 10⁸ | 7.4 GiB |
| LO | 1.283 / 1.525 / 1.778 | 19.3 % | 1.53 × 10⁸ | 8.7 GiB |
| HI | 1.284 / 1.525 / 1.777 | 12.8 % | 1.06 × 10⁸ | 6.0 GiB |

Wall times per input (one process, 4 numba threads): U 469 s (two scans reused, DECISION 13-20), SCH 1773 s, P5514 1732 s, LO 2032 s, HI 1560 s. N stopped after 486 s.

## 8. Figure

`round13/results/fig_partA.png` (one column, 8 pt).

**Caption:** "Information of the designed table (filled) and of the matched constant SoD (open) against the full-data information, both relative to the uniform input U, σ/120, budget B = 316.89. The matched SoD is shown for six inputs; the designed table only for inputs whose design finished (N: none, its scan stopped on a conservation assertion). What to look at: table points on the dotted line y = x mean that the designed table loses the same amount of information for every input (separation). The matched-SoD points are shown for contrast."

All five table points lie within 0.019 nat of y = x: U 0, P5514 −0.016, SCH +0.013, LO −0.007, HI +0.019. The matched-SoD points lie between −0.254 (LO) and +0.722 (P5514) from the line: SCH +0.650, HI +0.565, N +0.053, U 0. The LO and HI table points coincide (x = −0.974); their SoD points are 0.819 nat apart.

## 9. DECISION list (continuing Part 0's 13-1 … 13-16)

- **DECISION 13-17 (your decision).** Inputs run in sequence, one process at a time. `Stats` is computed only for T\*, after the design object and its caches are released; for an input where `Stats` would not fit, the shape statistics are skipped. In the event no input was skipped (§7). This replaces DECISION 13-7's "`Stats` for every scan table".
- **DECISION 13-18.** Memory rule for `Stats`: the estimated peak, 64 B × (nodes visited by the callback) + 1 GiB, must not exceed MemAvailable. The 64 B comes from the Part 0 probe, which measured 59.8 B. The estimates were 7.3–10.1 GiB against 15.1 GiB available.
- **DECISION 13-19.** Grids for the scoring (written in `predictions_partA.md` before running). Derived quantities and H1–H3 use σ/120. The point predictions use σ/24. The certificate gap uses σ/24, from scan j1, before any third scan.
- **DECISION 13-20.** U reuses the two completed scans of gate G0 through the checkpoint: the scan jsons were copied, and the final tables are in the work directory. Those scans were already shown bitwise equal to Round 11. T₀ and T₁ were re-evaluated (σ/24, σ/120, backward).
- **DECISION 13-21.** Rank agreement: values within 0.1 nat are tied. Two orders agree iff all pairs have the same tie-aware relation. Discordant pairs and mixed-tie pairs are listed. The printed orders chain ties for display only.
- **DECISION 13-22 (your decision).** In `dp13.Design11.__init__`, a kernel window of W88 − 1 nodes is accepted, and any other width stops. The kernel row keeps length W88, with the unused last entry exactly 0.0. The u / ndtr / φ caches use the fixed-length range starting at KA − XC. Per-step assertions are listed in §6.1, the neutrality checks for U are (a) and (b), and the end-to-end proof is the forward/backward lines. The same fix covers PAR in Part B.
- **DECISION 13-23.** Constant SoD at Δ\*: σ/24 through the accepted operator (`evaluator_fast`); σ/120 through `eval13.evaluate_refined(ConstRows(480))`, the path that G2 checked.
- **DECISION 13-24.** `Stats` uses γ from the j0 tangent (constant SoD at Δ\*), as in Round 11's `post11`.
- **DECISION 13-25.** The figure uses σ/120 values. Points of one series closer than 4 % of the axis span share one label. N appears with its matched SoD only.
- **DECISION 13-26.** A worker restart during LO's first scan (between 13:25 and 13:58) killed the first sequence. The sequence was relaunched after DECISION 13-22 in the order SCH, P5514, LO, HI, N. LO restarted from its saved matched SoD, since its first scan had not been checkpointed. `partA_LO.log` contains both attempts.
- **DECISION 13-27.** A stopped input is committed (partial json, log, stdout) and the sequence goes on with the next input.
- **DECISION 13-28.** Summary with a missing input:
  - c_sod, the matched-SoD rank comparison and the constant-SoD contrast use all six inputs;
  - quantities that need a table use the five inputs that have one;
  - the verdict is reported as "incomplete";
  - prediction lines that need every input are "undetermined" unless the available inputs already make them a miss.
- **DECISION 13-29.** For N, no tolerance was changed and no fallback was used. The relaxed-tolerance forwards in `diag_massdev.py` are a diagnostic and do not enter any result.

**Needs your decision before N can be completed (it also matters for PAR in Part B, whose grid is larger still, G = 49518).** Options:
- (a) Raise the σ/24 conservation tolerance, e.g. to 1e-11, for every input. The five finished inputs are unaffected: their largest deviation over every σ/24 forward is 9.1e-13 < 1e-12, and the tolerance only decides whether a run stops. Then re-run N (≈ 45 min).
- (b) Use a tolerance that scales with the grid size.
- (c) Leave N out; the verdict stays "incomplete".

## 10. Not done

- N's designed table, I_table, c_tab, gain, gap and shape statistics. N's I_full, constant SoD at Δ\* and matched SoD are complete.
- The §6 verdict over six inputs.
- Part B (as instructed).
- Monte Carlo cross-checks of the Part A tables (not asked for in Part A).
