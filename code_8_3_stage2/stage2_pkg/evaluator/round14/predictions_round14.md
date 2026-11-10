# Round 14 predictions (written and committed before any computation)

Source: `reports_CN/prompt_round14_one_parameter_rule_EN.md` §6 (verbatim below). At the time of this commit no round-14 code
exists and no round-14 number has been produced. The reference values of §2 (1a, 1b) come from a computation made outside the
repository, as the prompt says.

```
- 1a, 1b reproduce to 1e-9; fit_rule(B_nom) = (37, 13).
- 1c: |z| <= 3 on all three quantities.
- 1d: e_C in [1.56, 1.60]; continuous-rule log det within 0.05 nat of 207.498.
- 2a: delta in [0.02, 0.08] sigma at every budget; e_eff about 1.88, 1.70, 1.58, 1.45, 1.35 (each +-0.04) for B = 160, 240, 317, 400, 480;
      Delta I over matched SoD in [9, 12] nat at every budget; rule below the class-W optimum by 1.2 to 2.0 nat at every budget.
- 2b: range of c_rule <= 0.1 nat; I_rule - I_table in [-0.1, +0.3] nat for every input; delta in [0.02, 0.08] sigma.
Scoring: a line is a hit iff every stated number falls in its range.
```

## Scoring conventions

These are fixed now, before running. The prompt leaves the grid and some references open for these lines; this is DECISION 14-1.

1. **Line "1a, 1b".** Hit iff all of the following hold:
   - every 1a value (E[N], S, log det J for n = 36, 37, 38, 40, σ/24) and every 1b value (E[N], S, log det J of R(37, 13) on σ/24 and σ/120) is within 1e-9 relative of the prompt's reference;
   - cond(J), given to 4 decimals, agrees at that precision (abs(diff) ≤ 5e-5);
   - `fit_rule(B)` returns (37, 13).
2. **Line "1c".** Hit iff abs(z) ≤ 3 for E[N], S and log det J. Each z compares the σ/120 value with the Monte Carlo mean of the lattice rule R(37, 13): standard error of the mean for E[N] and S, batch standard error for log det J, as in `eval11.monte_carlo`.
3. **Line "1d".** Hit iff both hold:
   - e_C, from linear interpolation of the Monte Carlo E[N] in e, is in [1.56, 1.60];
   - abs(I_cont − 207.498) ≤ 0.05 nat, where I_cont is the log det J of the extra Monte Carlo at e_C rounded up to 0.005 σ.
4. **Line "2a".** Hit iff all of the following hold for each of B = 160, 240, 316.892…, 400, 480:
   - δ = e_eff − e_1 ∈ [0.02, 0.08] σ;
   - e_eff within ±0.04 σ of 1.88, 1.70, 1.58, 1.45, 1.35 respectively;
   - ΔI = I_rule − I_SoD ∈ [9, 12] nat, both on σ/120 (rule: `fit_rule(B)`; SoD: `common13.matched_sod(B)`);
   - f − I_rule ∈ [1.2, 2.0] nat, where f is round 10's `solve(prob, B − 1)` value (the class-W optimum) and I_rule is on σ/120. u_cert is reported alongside.
5. **Line "2b".** Hit iff all of the following hold over the six inputs U, SCH, P5514, LO, HI, N at the nominal budget (U from Part 1):
   - range of c_rule = I_rule − I_full ≤ 0.1 nat, with I_rule on σ/120;
   - I_rule − I_table ∈ [−0.1, +0.3] nat for every input, with I_table the σ/120 log det J of round 13's T\*;
   - δ ∈ [0.02, 0.08] σ for every input.

The questions Q1–Q5 of prompt §4 use the same quantities:
- Q2's "meets the budget" means the Monte Carlo E[N] at e_C rounded up is ≤ B (point estimate); (E[N] − B)/SE is reported alongside.
- Q2's lattice reference is the σ/120 log det J of R(37, 13).
- Q3 uses all five budgets and the six inputs.
- Q4 uses σ/120.
- Q5's "within 0.3 nat" means abs(I_rule − I_table) ≤ 0.3 on σ/120.
