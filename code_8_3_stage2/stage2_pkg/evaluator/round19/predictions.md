# Round 19 predictions (written and committed before any round-19 computation)

Source: `reports_CN/prompt_round19_two_stage_rule_EN.md` §7 (verbatim below). At the time of this commit no round-19
code exists and no round-19 number has been produced. No new number was computed from archived results either.

## 1. Predictions of the prompt author (verbatim)

```
These are the predictions of the prompt author. You may add your own in a separate section of the same file, labelled as yours.
- K1 = 2500: net gain 6.0 nat (range 5.3–6.6). Basis: if the information matrices of the two stages are roughly proportional, 9*ln(0.5*(1+exp(10.30/9))) = 6.55, minus the losses from model error and feedback.
- K1 = 1250: net gain 8.1 nat (range 7.3–8.6).
- K1 = 3750: net gain 3.7 nat (range 3.2–4.0).
- Control (true model in stage 2, K1 = 2500): 6.4 nat.
- Total transmissions: mean between 312 and 316; about 10% of replications above B (range 3%–25%).
- The budget is most at risk for K1 = 1250: its fraction above B is higher than for K1 = 2500.
- Final estimate: largest bias below 0.3 CR standard deviations; mean-square error at most 5% above the CR bound.
```

## 2. Scoring conventions (fixed now, before running)

All ranges are closed intervals. "Above" and "below" are strict; "between" and "at most" are inclusive.

**Common definitions.**
- `J(config) = sigma^-2 sum_k wbar_k h_k h_k^T`, with `wbar_k` the mean over the configuration's replications of
  `c_k = sum(dr.cfun((lo_k - x_k)/sigma, (hi_k - x_k)/sigma))`, using the true `x_k` and the realised window. Step 1 is the
  forced first transmission, `c_1 = 1`. This equals the mean over replications of the per-run `J_cond`.
- **Net gain** of a configuration = `log det J(config) - log det J(baseline)`. The baseline is constant SoD throughout,
  evaluated on the **same replications** (same noise sequences). Primary: replications 0..999. Secondary and control:
  replications 0..299.
- The 95 % interval is the percentile interval of a paired bootstrap over replications. There are 2000 resamples (seed
  194000000), and each resample uses the same indices for the configuration and for the baseline.
- **Total transmissions** N counts every transmission, the forced first one included. `N_1` covers steps 1..K1 and `N_2`
  covers steps K1+1..K. "Above B" means `N > B = 316.8921168493721`, i.e. `N >= 317`.
- **Largest bias:** `max_i |mean_r(theta_hat_r,i - theta*_i)| / sqrt((J^-1)_ii)`. Here `J` is the configuration's own
  `J(config)`, and the statistics run over all replications, using the decided estimate of each one (fallbacks included).
- **MSE against the CR bound:** `M-hat = (1/R) sum_r e_r e_r^T` with `e_r = theta_hat_r - theta*`, `D-hat = -log det M-hat`,
  and `I = log det J(config)`. Then `log det(M-hat J) = I - D-hat`.

**Lines of §1.**
1. **K1 = 2500.** Hit iff the net gain (primary, 1000 replications) lies in [5.3, 6.6].
2. **K1 = 1250.** Hit iff the net gain lies in [7.3, 8.6].
3. **K1 = 3750.** Hit iff the net gain lies in [3.2, 4.0].
4. **Control.** The prompt gives no range. Hit iff the point estimate rounds to 6.4 at one decimal, i.e. lies in
   [6.35, 6.45). Whether 6.4 lies inside the control's 95 % bootstrap interval is reported as well, but is not the score.
5. **Total transmissions (primary).** Hit iff mean N lies in [312, 316] **and** the fraction of replications with N > B
   lies in [0.03, 0.25]. The two parts are also reported separately.
6. **Budget risk.** Hit iff the fraction with N > B for K1 = 1250 (300 replications) is **strictly greater** than that for
   K1 = 2500 (primary, 1000 replications). The comparison on the paired replications 0..299 is reported too, but is not the
   score.
7. **Final estimate (primary, all 1000 replications).** Hit iff the largest bias is below 0.3 **and** the MSE is at most 5 %
   above the CR bound. The latter is read as the per-parameter geometric mean of the variance ratio:
   `exp((I - D-hat)/9) <= 1.05`, i.e. `I - D-hat <= 9 ln 1.05 = 0.4391 nat`. Two other readings are reported but not scored:
   the standard-deviation reading `exp((I - D-hat)/18) <= 1.05`, and the accepted-only replications.

The headline tally runs over these seven lines.

## 3. My own predictions (Claude; labelled as mine, not the prompt author's)

These come from reasoning only. I ran no round-19 code and did no new computation on archived results. The reasoning:
- **Ideal two-stage value.** Minkowski's determinant inequality makes the prompt author's proportional value (6.55 at
  K1 = 2500) a *lower* bound on the ideal two-stage value. The excess for J_SoD and J_rule should be small (~0.03 nat).
- **Losses**, scaled by the share of stage 2 and by the per-parameter ratio J_rule / J_two-stage:
  - the 3-transmission margin in q_2 (~0.08 nat at K1 = 2500);
  - the feedback's count fluctuation (~0.07 nat);
  - the model error. Round 15 found VF(100, 3) costs 0.28 nat at c = 2, and a stage-1 estimate from half the record is
    roughly c = 2. This gives ~0.2 nat at K1 = 2500.
- **Budget.** Model error raises the rule's transmission rate at a given e. The proportional feedback then settles above
  plan, by `2 q_2 T * Delta e`. That raises the count most at K1 = 1250.

| # | quantity | my prediction |
|---|---|---|
| C1 | net gain, K1 = 2500 | 6.2 nat, range [5.8, 6.5] |
| C2 | net gain, K1 = 1250 | 7.9 nat, range [7.4, 8.3] |
| C3 | net gain, K1 = 3750 | 3.6 nat, range [3.3, 3.8] |
| C4 | control (true model in stage 2, K1 = 2500) | 6.4 nat, range [6.2, 6.6]; control minus primary on the paired replications 0..299 in [0.05, 0.40] nat |
| C5 | total transmissions, primary | mean in [314.0, 315.5]; sd in [1.6, 2.6]; fraction N > B in [0.08, 0.25] |
| C6 | ordering of the fraction N > B | K1 = 1250 > K1 = 2500 > K1 = 3750 (strict) |
| C7 | CML non-acceptance, primary | stage 1: between 1 and 15 of 1000; final joint estimate: at most 25 of 1000 |
| C8 | RMS prediction error of theta_tilde (median over replications, units of sigma) | K1 = 2500 in [0.12, 0.25]; medians ordered K1 = 1250 > K1 = 2500 > K1 = 3750 |
| C9 | failure cases, primary | e_k reaches 0.5 or 4 in at most 2 of 1000 replications; abnormal stage-2 count in 0 replications |
| C10 | final estimate, primary | largest bias below 0.3; I − D-hat over all replications in [0.3, 1.0] nat (so §1 line 7 misses under the variance reading); I − D-hat over accepted-only replications in [0.15, 0.6] nat |
| C11 | empirical measure, primary | the paired difference of D-hat (two-stage minus baseline; the round-8b measure) lies within ±0.5 nat of the log det J net gain |

Scoring of my lines: a line is a hit iff every number in it falls in its range.

## 4. Procedure fixed before running

- **Code.** New code only in `evaluator/round19/`. The trigger windows are those of `round18/acc18.window`:
  - SOD: `r ± Delta*`.
  - RULE / RULE_FB: `min(x - e s, r - s)`, `max(x + e s, r + s)`, with the feedback in the arithmetic form of
    `round15/mismatch15.run_job`.
  - In stage 2, `x_k` is replaced by the prediction `xbar_k`, and the plan is generalised from `1 + q_t (k - 1)` to
    `N_1 + q_2 (k - K1)`. With K1 = 1 and N_1 = 1 this is the existing form exactly.
- **Information.** `c_k` comes from `dr.cfun` with the true `x_k`, as in `acc18.batch_A`.
- **Stage-1 estimate.** `theta_tilde` is `estimator_r8b.estimate_stage1(stage-1 data, 191000000 + i)`, unchanged:
  - LS initial value;
  - one restart;
  - the round-8b Newton-decrement acceptance;
  - fallback `theta_LS`, which is used as `theta_tilde` when the run is not accepted. Such runs are counted and listed.
- **Final estimate.** `estimator_r8b.estimate_joint([whole-record data], theta_tilde, 192000000 + i)`, unchanged: start at
  `theta_tilde`, one restart, the same acceptance, fallback `theta_tilde`. The record's windows are reconstructed by the
  receiver:
  - `r_k` from the record;
  - SoD windows for k <= K1;
  - rule windows for k > K1, with `xbar` from `theta_tilde` and `e_k` from the counts in the record.
  - The baseline's final estimate is the same function on its own whole SoD record, from the same `theta_tilde`. The
    baseline's stage-1 record is identical to the two-stage one.
- **Seeds.**
  - Noise of replication i: `np.random.default_rng(190000000 + i).standard_normal(K)`. Value 0 is the noise of z_1 and
    value p that of step p + 1. The same noise is used for every configuration and for the baseline.
  - Checks (a) and (b): 10 000 trajectories, noise seeds `193000000 + i`, i = 0..9999, in 10 batches of 1000. The SE of
    log det is the batch SE (round-18 convention); the SE of E[N] is sd(N)/sqrt(10 000).
  - Bootstrap: seed 194000000, 2000 resamples.
- **Stop rule.** If check (a) or check (b) fails, the run stops and nothing else is computed.
