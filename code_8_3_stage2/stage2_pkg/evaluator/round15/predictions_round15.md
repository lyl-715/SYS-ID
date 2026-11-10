# Round 15 predictions (written and committed before any computation)

Source: `reports_CN/prompt_round15_rule_with_estimated_model_EN.md` §7 (verbatim below). At the time of this commit no
round-15 code exists and no round-15 number has been produced.

```
- Check 1 reproduces to 1e-9; check 2 passes.
- Q1: fraction over budget for V0 between 30 % and 65 % at every level; 5-95 % range of E[N] - B has half-width in
      [20, 50] at c = 1, [30, 70] at c = 2, [40, 100] at c = 4.
- Q2: yes at c = 1 and c = 2.
- Q3: the complying margin is Delta >= 0.10; its median Delta I at c = 2 is at least 0.8 nat below the true-model V0.
- Q4: VF(100, 3) complies; some complying feedback setting has median Delta I at c = 2 within 0.4 nat of the true-model V0;
      true-model cost of VF(T, 0) is <= 0.15 nat for T >= 100.
- Q5: VF* above VM* by at least 0.3 nat at c = 1 and c = 2.
- Q6: median sd of N below 4 for T <= 100 and between 16 and 21 for V0.
- Q7: VF* keeps <= 5 % over budget up to eps = 0.5; at the 2 % gross error E[N] > 2 B for every variant.
Scoring: a line is a hit iff every stated number falls in its range.
```

## Scoring conventions

These are fixed now, before running (DECISION 15-1). Common definitions:
- "Level" means c ∈ {1, 2, 4}; each level has 40 models of the main set.
- "E[N] > B" uses the Monte Carlo mean of the job (point estimate).
- Percentiles over models use `numpy.percentile` with linear interpolation.
- "True-model V0" is the V0 job with δ = 0, R = 20000, and its ΔI = log det J − 197.20097668224076.

1. **Line "Check 1, check 2".** Hit iff both hold:
   - check 1: for all 12 reference models, E[N] and log det J are within 1e-9 relative of the table, and the rms and max prediction errors agree at the printed 6 decimals (abs(diff) ≤ 5e-7);
   - check 2: abs(z) ≤ 3 for E[N], S and log det J for each of the three models, with 0 clipped look-ups.
2. **Line Q1.** Hit iff, for V0:
   - the fraction of models with E[N] > B lies in [0.30, 0.65] at c = 1, 2 and 4;
   - the half-width (P95 − P5)/2 of E[N] − B over the 40 models lies in [20, 50] at c = 1, [30, 70] at c = 2 and [40, 100] at c = 4.
3. **Line Q2.** Hit iff abs(median ΔI(V0) − ΔI(true-model V0)) ≤ 0.5 nat at c = 1 and at c = 2. (c = 4 is answered in the report but is not part of the line.)
4. **Line Q3.** Hit iff all three hold:
   - at least one margin variant complies (prompt §4b);
   - VM(0.05) does not comply, so every complying margin has Δ ≥ 0.10;
   - for VM\*, ΔI(true-model V0) − median ΔI(VM\*, c = 2) ≥ 0.8 nat.
5. **Line Q4.** Hit iff all three hold:
   - VF(100, 3) complies;
   - some complying VF setting has ΔI(true-model V0) − median ΔI(VF, c = 2) ≤ 0.4 nat;
   - ΔI(true-model V0) − ΔI(true-model VF(T, 0)) ≤ 0.15 nat for T = 100, 300 and 1000.
6. **Line Q5.** Hit iff median ΔI(VF\*) − median ΔI(VM\*) ≥ 0.3 nat at c = 1 and at c = 2.
7. **Line Q6.** Hit iff both hold:
   - the median (over the 40 models) sd of N is < 4 for every VF setting with T ∈ {50, 100} (m = 0, 3, 6) at c = 1, 2 and 4;
   - the median sd of N of V0 lies in [16, 21] at c = 1, 2 and 4.
8. **Line Q7.** Hit iff both hold:
   - for VF\*, the fraction of the 40 scaled models with E[N] > B is ≤ 0.05 at ε = 0.5;
   - for both gross-error models (+2 % and −2 %), E[N] > 2B for V0, VM\* and VF\*.
