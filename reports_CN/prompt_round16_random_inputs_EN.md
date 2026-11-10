# Round 16 — random inputs: is the information lost by the trigger independent of the input?

## 0. Why, and rules

Rounds 13 and 14 found, on six hand-picked inputs, that the information lost by the designed trigger relative to full sampling hardly depends on the input (range 0.035 nat for the tables, 0.117 nat for the one-parameter rule), whereas for constant-threshold SoD at the same budget it does (0.976 nat). Six chosen inputs support the statement but cannot give a distribution or a worst case. This round repeats the comparison on 200 **random** inputs, for the one-parameter rule against constant SoD at the same budget, by Monte Carlo without a grid.

- Create branch `round16-random-inputs` from the head of `round14-one-parameter-rule`; state that commit in the report. Push only to the new branch. Rounds 4–15 are read-only; new code only in `evaluator/round16/`. This round does not depend on Round 15.
- First commit: this prompt into `reports_CN/`. Second commit: `round16/predictions_round16.md` (§7) before any computation.
- No DP, no grid evaluator. Everything is Monte Carlo of the continuous rule and of constant SoD. The frequencies `dr.w`, `a0 = 1`, noise `s`, `K = 5000` and budget `B = 316.8921168493721` are unchanged.
- Assertion failures stop the run. Every choice left open is a `DECISION 16-x` line. Delete any empty folders created under `round4/`.

## 1. Inputs

An input is `(p, χ)`: powers `p_m = a_m²` on the four frequencies with `Σ p_m = 1`, phases `χ_m`. Output and regressors from `dr.output(a, chi)` and `dr.regressors(a, chi, dr.t)` with `a = sqrt(p)`.

- **Random set (200).** `rng = np.random.default_rng(20261113)`; repeat until 200 are accepted: `p = rng.dirichlet(np.ones(4))`, `chi = rng.uniform(0, 2π, 4)`; accept iff `min(p) ≥ 0.02` (a frequency with almost no power makes its two parameters unidentifiable for every trigger; the bound is fixed here, DECISION 16-1). Report how many draws were rejected.
- **Anchors (6).** `U`, `SCH`, `P5514`, `LO`, `HI`, `N` as defined in `round13/inputs13.json` and its sources.

For every input record: `p`, `χ`, output range `(max x − min x)/σ`, total variation `Σ_k |x_{k+1} − x_k| / σ`, largest per-step motion in σ, and `I_full = log det(HᵀH/σ²)`.

## 2. What is computed for each input

All Monte Carlo runs are vectorised over trajectories as `eval11._mc_batch` (noise drawn step by step, initial `r = x_1 + σ·noise`, first step always transmitted), with `R = 2000` trajectories and the **same noise seed `20261113` for every run and every input** (common random numbers). Information: `W_k` = mean over trajectories of `dr.cfun((lo − x_k)/σ, (hi − x_k)/σ)`, `W_1 = 1`, `J = σ⁻² Σ_k W_k h_k h_kᵀ`.

**2a. The one-parameter rule.** Windows `lo = min(x_k − eσ, r − σ)`, `hi = max(x_k + eσ, r + σ)` (the model is the true one, so the prediction is `x_k`).
1. Run `e ∈ {1.52, 1.56, 1.60, 1.64}`; if `B` is not bracketed by the four `E[N]`, extend the list by steps of 0.04 on the side needed until it is.
2. Between the two bracketing values interpolate linearly in `e`: `e_fit` where `E[N] = B`, and `I_rule` = the `log det J` at that point (linear interpolation of `log det J` against `E[N]` between the same two runs). `S_rule` likewise.

**2b. Constant SoD at the same budget.** Windows `lo = r − Δ`, `hi = r + Δ`.
1. Run `Δ/σ ∈ {12, 16, 20, 24, 28, 34}`; extend by steps of 4 on the side needed until `B` is bracketed; then run the two points at one third and two thirds of the bracketing interval.
2. Between the two closest bracketing values interpolate as in 2a: `Δ_fit`, `I_SoD`, `S_SoD`.

**2c. Derived.**
- `c_rule = I_rule − I_full`, `c_sod = I_SoD − I_full` — the information lost relative to full sampling;
- `gain = I_rule − I_SoD`;
- information per transmission `(S − 1)/(B − 1)` for both.

## 3. Checks

1. **Anchors against the archives.** For the six anchors compare `c_rule` with Round 14 (part 2b, σ/120 lattice rule; for `U` part 1b) and `c_sod` with Round 13 Part A (matched SoD, σ/120): report the differences; pass if every `|difference| ≤ 0.06` nat. Also compare `e_fit` with Round 14's `e_eff` and `Δ_fit/σ` with Round 13's `M*/24`.
2. **Noise floor.** For input `U` repeat 2a and 2b with ten other noise seeds (`20261114 … 20261123`): standard deviation of `c_rule`, `c_sod`, `e_fit`, `Δ_fit` over the ten. This is the Monte Carlo noise of a single input's values without common random numbers; say so.
3. **Pair check.** For `LO` and `HI`: `c_rule(LO) − c_rule(HI)` against the difference of Round 14's rule values for these two inputs, and `c_sod(LO) − c_sod(HI)` against Round 13's matched-SoD difference (−0.819); report both differences.
4. Every run: `0 ≤ W_k ≤ 1`, `N ≥ 1`, `log det J` finite; every interpolation uses two runs that bracket `B`.

If check 1 fails, stop and report: the Monte Carlo route would then not be measuring the same quantities as the earlier rounds.

## 4. What to report

**4a. Distribution over the 200 random inputs** of `c_rule`, `c_sod`, `gain`, `e_fit`, `Δ_fit/σ`: mean, standard deviation, minimum, 5 %, median, 95 %, maximum. For `c_rule` and `c_sod` also the same statistics after subtracting the value of input `U`.

**4b. The extremes.** The five inputs with the lowest and the five with the highest `c_rule`, and the same for `c_sod`: their `p`, `χ`, output range, total variation.

**4c. What the spread goes with** (descriptive only). Spearman correlations of `c_rule` and of `c_sod` with: total variation, output range, `max p`, `min p`, `I_full`.

**4d. Phase-only and power-only parts.** From the random set build two derived sets and run them the same way: (i) the first 50 phase vectors with uniform power `p = (¼, ¼, ¼, ¼)`; (ii) the first 50 power vectors with zero phases. Report the 4a statistics of `c_rule` and `c_sod` for each.

**4e. One figure** (`fig_round16.png`, one column, font ≥ 8 pt): two histograms on a common axis, `c_rule − median(c_rule)` and `c_sod − median(c_sod)` over the 200 random inputs (bins of 0.05 nat), with the six anchors marked as ticks. Caption states what to look at: the width of the two histograms.

## 5. Questions the report must answer (fixed before running)

- **Q1** How wide is the distribution of `c_rule` over random inputs (standard deviation; 5–95 % range; worst case)?
- **Q2** How wide is that of `c_sod`?
- **Q3** Is the 5–95 % range of `c_rule` at most one third of that of `c_sod`?
- **Q4** Is `e_fit` the same for all inputs to within ±0.03 σ (5–95 %)?
- **Q5** Is `gain ≥ 8.5` nat for at least 95 % of the inputs? What are its minimum and maximum?
- **Q6** Were the six hand-picked inputs representative: where do their `c_rule` and `c_sod` fall in the random distributions (percentiles)?
- **Q7** Is the spread of `c_sod` mainly a phase effect or a power effect (4d)?

Answer each with numbers. No causal statements; 4c is a description, not an explanation.

## 6. Not in this round

The designed tables (they need a DP per input); other frequencies, other total power, other systems, other budgets; model mismatch; any input with `min p < 0.02`.

## 7. Predictions (commit before running)

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

## 8. Outputs

- `round16/random16.py`, `round16/run_round16.py`, `round16/results/round16.json` (one record per input with every run), `fig_round16.png`, logs.
- `reports_CN/report_round16_CN.md` (+ EN copy): (1) answers to Q1–Q7, one or two lines each; (2) the distribution table; (3) extremes; (4) correlations; (5) phase-only and power-only sets; (6) anchors and noise floor; (7) predictions hit/miss; (8) checks; (9) DECISION list; (10) wall times; (11) not done.
- Runtime guide: one run of 2000 trajectories takes one to two seconds; about 13 runs per input, 306 inputs in total (200 + 6 + 100 derived) plus the noise-floor repeats → about 4300 runs, 25–35 min on four processes.
