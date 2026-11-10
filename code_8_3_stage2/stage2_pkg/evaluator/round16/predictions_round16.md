# Round 16 predictions (written and committed before any computation)

Source: `reports_CN/prompt_round16_random_inputs_EN.md` §7 (verbatim below). At the time of this commit no round-16 code
exists and no round-16 number has been produced.

```
- Check 1 passes (all six anchors within 0.06 nat on both quantities).
- Q1: 5-95 % range of c_rule <= 0.30 nat; standard deviation <= 0.10 nat; worst case within 0.45 nat of the median.
      (Wider than the 0.117 nat of the six inputs: a uniform draw of the powers gives more unequal splits than any of the six.)
- Q2: 5-95 % range of c_sod in [0.6, 1.8] nat.
- Q3: yes.
- Q4: yes; median e_fit in [1.565, 1.590].
- Q5: yes; median gain in [9.2, 10.0] nat.
- Q6: U lies in the lowest 10 % of c_sod (zero phases are an unfavourable case for constant SoD); the anchors' c_rule all lie
      between the 2nd and 98th percentiles.
- Q7: in the phase-only set the 5-95 % range of c_sod is at least 4 times that of c_rule.
Scoring: a line is a hit iff every stated number falls in its range.
```

## Scoring conventions

These are fixed now, before running (DECISION 16-2). All quantities are the Monte Carlo values of prompt §2 (R = 2000, common
noise seed 20261113, linear interpolation to E[N] = B); the 200 random inputs are the "random set"; the anchors and the
derived sets of §4d are not part of it.

- Percentiles are `numpy.percentile` with its default (linear) interpolation. "5-95 % range" of a quantity = P95 - P5 over the
  set. "Standard deviation" uses ddof = 1. "Median" is `numpy.median`.
- The percentile rank of a value v within the random set is 100 (#{c < v} + 0.5 #{c = v}) / 200.

1. **Line "Check 1".** Hit iff, for each of U, SCH, P5514, LO, HI, N:
   abs(c_rule - c_rule(Round 14)) <= 0.06 nat and abs(c_sod - c_sod(Round 13 Part A matched SoD, sigma/120)) <= 0.06 nat.
   Round 14 reference: `round14/results/round14.json -> input_table.<name>.c_rule` (sigma/120 lattice rule; U from Part 1b).
   Round 13 reference: `round13/results/partA_<name>.json -> derived.c_sod` (sigma/120).
2. **Line "Q1".** Hit iff P95 - P5 of c_rule <= 0.30 nat, std(c_rule) <= 0.10 nat and max |c_rule - median(c_rule)| <= 0.45 nat.
   The parenthesis is not scored.
3. **Line "Q2".** Hit iff 0.6 <= P95 - P5 of c_sod <= 1.8 nat.
4. **Line "Q3".** Hit iff (P95 - P5)(c_rule) <= (1/3) (P95 - P5)(c_sod).
5. **Line "Q4".** Hit iff P95(e_fit) - median(e_fit) <= 0.03 sigma and median(e_fit) - P5(e_fit) <= 0.03 sigma (both sides of
   "within +-0.03 sigma"), and 1.565 <= median(e_fit) <= 1.590 sigma.
6. **Line "Q5".** Hit iff at least 190 of the 200 random inputs have gain = I_rule - I_SoD >= 8.5 nat, and
   9.2 <= median(gain) <= 10.0 nat.
7. **Line "Q6".** Hit iff the percentile rank of c_sod(U) within the random set is <= 10, and the percentile rank of c_rule of
   each of the six anchors within the random set is in [2, 98]. The anchors' values are this round's Monte Carlo values (same
   seed as the random set). The parenthesis is not scored.
8. **Line "Q7".** Hit iff, over the 50 phase-only inputs of §4d (i), (P95 - P5)(c_sod) >= 4 (P95 - P5)(c_rule).

Questions Q1-Q7 of prompt §5 use the same quantities and the same conventions.
