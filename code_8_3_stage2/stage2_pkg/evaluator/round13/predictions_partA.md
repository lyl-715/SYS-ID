# Round 13 Part A predictions (written and committed before any Part A computation)

Source: `reports_CN/prompt_round13_partA_input_invariance_EN.md` §7 (verbatim below). At the time of this commit no Part A
number has been produced. The only Part A input already evaluated is U: Part 0 gate G0 ran the U design, and it reproduces
Round 11 (I_table(U) = 207.35521730257756 on σ/24). The prompt's point predictions are built on that value
(c(U) = 207.355 − 219.009).

```
- H1 holds: S_table within ±3 % of S_table(U) for all six inputs.
- H2 holds: range of c_tab <= 0.5 nat.  Point predictions on sigma/24, from c(U) = 207.355 − 219.009 = −11.65:
  I_table ≈ 207.35 (P5514, SCH), 206.38 (LO, HI), 206.76 (N), each ±0.3.
- H3 holds.
- Constant SoD is not separable: range of c_sod >= 0.7 nat, and |c_sod(LO) − c_sod(HI)| >= 0.5 nat.
- Matched SoD: I_SoD(P5514) − I_SoD(U) ∈ [0.5, 0.9] nat (paper 1's +0.70 at nearly the same budget).
- Certificate gap <= 0.3 nat for every input after two scans.
- gain = I_table − I_SoD ∈ [9, 11.5] nat for every input.
Scoring: a line is a hit iff every stated number falls in its range.
```

## Scoring conventions

These are fixed now, before running. The prompt leaves the grid open for some lines; this choice is DECISION 13-19.

All derived quantities use σ/120, as Part A §4 item 5 says, unless a line says σ/24. Definitions:
- I_table is the σ/120 log det J of the best table T\*.
- I_SoD is the σ/120 log det J of the matched constant SoD.
- I_full = log det(HKᵀ HK / σ²).
- c_tab = I_table − I_full and c_sod = I_SoD − I_full.
- S is Σ_k w_k of the forward record on σ/120.

1. **H1 line.** Hit iff max over the six inputs of abs(S_table(i)/S_table(U) − 1) ≤ 0.03 (S of T\*, σ/120).
2. **H2 line.** Hit iff both of these hold:
   - range(c_tab) = max − min over the six inputs ≤ 0.5 nat (σ/120);
   - every σ/24 I_table point falls in its stated interval: P5514 and SCH in [207.05, 207.65]; LO and HI in [206.08, 206.68]; N in [206.46, 207.06].
3. **H3 line.** Hit iff all three hold (σ/120): abs(I_table(SCH) − I_table(U)) ≤ 0.2; abs(I_table(P5514) − I_table(U)) ≤ 0.2; abs(c_tab(LO) − c_tab(HI)) ≤ 0.5.
4. **Constant-SoD line.** Hit iff range(c_sod) ≥ 0.7 nat and abs(c_sod(LO) − c_sod(HI)) ≥ 0.5 nat (σ/120, matched SoD).
5. **Matched-SoD line.** Hit iff 0.5 ≤ I_SoD(P5514) − I_SoD(U) ≤ 0.9 nat (σ/120).
6. **Certificate line.** Hit iff, for every input, the gap after two scans is ≤ 0.3 nat. The gap is UB′(j1) − max(I(T₀), I(T₁)) on σ/24, with UB′(j1) = I(T₀) − 9 + min_λ [dual_j1(λ) + 0.25 λ]. A third scan, if any, does not count for this line.
7. **Gain line.** Hit iff 9 ≤ I_table − I_SoD ≤ 11.5 nat for every input (σ/120).

The verdict rules H1–H3 and the resolution condition are those of prompt §6. They use the same quantities as lines 1–3. The resolution condition is the final certificate gap ≤ 0.5 nat, after a third scan where one was run.
