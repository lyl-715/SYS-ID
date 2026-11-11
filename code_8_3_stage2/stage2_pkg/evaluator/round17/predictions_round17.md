# Round 17 predictions (written and committed before any computation)

Source: `reports_CN/prompt_round17_absolute_diagnostics_EN.md` §6 (verbatim below). At the time of this commit no round-17 code
exists and no round-17 number has been produced. Part C is on hold and not predicted here.

```
- All reproductions pass.
- Q1/Q3: rule R(37,13): mu_min in [0.255, 0.270], mu_max in [0.283, 0.297], cond in [97, 101].
         T': mu_min >= 0.26, mu_max <= 0.31, cond in [97, 101].
         fixed residual threshold: all nine mu within 0.001 of 0.327.
         class-W optimum: all mu in [0.30, 0.37].
- Q2: larger in the worst direction: mu_min(rule)/mu_min(SoD) >= 5, against a ratio of geometric means of about 3.1.
- Per-parameter standard-deviation ratios: rule all in [1.80, 2.00]; constant SoD all in [2.4, 4.6].
- Q4: monotone; I(0.25 sigma) - I(1 sigma) in [0.05, 0.15] nat; I(1 sigma) - I(2 sigma) in [0.15, 0.25] nat (both grids and the Monte Carlo).
Scoring: a line is a hit iff every stated number falls in its range.
```

## Scoring conventions

These are fixed now, before running (DECISION 17-1).
- μ are the generalised eigenvalues of J v = μ J_full v; cond is the 2-norm condition number.
- Part A uses the σ/24 matrix of each design; the fixed residual threshold and class W are matrices of round 10's model.

1. **Line "All reproductions pass".** Hit iff all of the following hold:
   - every archived log det J of prompt §1 reproduces to 1e-9 relative;
   - cond(J_full) and the constant-SoD μ_min, μ_max and cond reproduce to 1e-9 relative;
   - the geometric-mean identity holds to 1e-10 for every design;
   - M_min = 24 reproduces Round 14's R(37, 13) bit for bit;
   - round 10's f and u_cert reproduce exactly.
2. **Line Q1/Q3.** Hit iff every stated range holds:
   - rule R(37, 13): μ_min ∈ [0.255, 0.270], μ_max ∈ [0.283, 0.297], cond ∈ [97, 101];
   - T′: μ_min ≥ 0.26, μ_max ≤ 0.31, cond ∈ [97, 101];
   - fixed residual threshold: abs(μ_i − 0.327) ≤ 0.001 for all nine;
   - class-W optimum: all nine μ_i ∈ [0.30, 0.37].
3. **Line Q2.** Hit iff both hold:
   - μ_min(rule)/μ_min(SoD) ≥ 5;
   - the ratio of geometric means, exp((I_rule − I_SoD)/9), rounds to 3.1 at one decimal, i.e. lies in [3.05, 3.15).
4. **Line "standard-deviation ratios".** Hit iff all nine sqrt(diag(J⁻¹)/diag(J_full⁻¹)) of the rule lie in [1.80, 2.00] and all nine of the constant SoD lie in [2.4, 4.6].
5. **Line Q4.** "Monotone" means that I increases strictly at every step as M_min goes 48 → 36 → 24 → 18 → 12 → 6. Hit iff, separately on σ/24, on σ/120 and for the continuous Monte Carlo (log det J interpolated to E[N] = B), all three hold:
   - I is monotone in that sense;
   - I(6) − I(24) ∈ [0.05, 0.15] nat;
   - I(24) − I(48) ∈ [0.15, 0.25] nat.
