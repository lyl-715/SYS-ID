# Round 18 predictions (written and committed before any computation)

Source: `reports_CN/prompt_round18_accumulated_information_EN.md` §6 (verbatim below). At the time of this commit no
round-18 code exists and no round-18 number has been produced.

```
- Part A: sd of per-run log det: RULE in [0.04, 0.09], SOD in [0.20, 0.33]; I - mean: RULE <= 0.005, SOD in [0.02, 0.05].
          Value of the non-transmissions: RULE in [12.8, 13.8] nat, SOD in [3.3, 4.1] nat, PRED in [14.0, 15.5] nat.
- Part B: LS under RULE: largest bias above 3 Cramer-Rao standard deviations; D-hat more than 10 nat below I.
          LS under SOD: largest bias above 0.5 standard deviations.
          CML under RULE and PRED: largest bias below 0.2 standard deviations; D-hat within 0.3 nat of I; at most 2 % of runs not accepted.
- Part C: I(C1) - I(C0) in [0.00, 0.20] nat; I(C2) - I(C1) in [-0.20, +0.05] nat.
Scoring: a line is a hit iff every stated number falls in its range.
```

## Scoring conventions

Fixed now, before running (DECISION 18-1). All ranges are closed intervals; "above" and "below" are strict, "at most" and
"within" are inclusive.

**What a "line" is.** The block has three bullets (Part A, Part B, Part C) written on six physical lines. Each physical line
is scored as one line: A1, A2, B1, B2, B3, C. A bullet is a hit iff all its physical lines are hits. The headline tally is
over the six lines; the three-bullet tally is reported alongside.

**Common definitions.**
- `J_cond`, `J_sent`, `I` are as in prompt §2. `I = log det(mean J_cond)` over the R = 10^4 Part A trajectories of the trigger.
- Per-run `log det` is `log det J_cond` of one trajectory. "sd" is the sample standard deviation (ddof = 1) over the 10^4
  trajectories; "mean" is their arithmetic mean.
- Cramér–Rao standard deviation of parameter i: `sqrt((J^-1)_ii)` with `J` = the Part A mean `J_cond` of the same trigger.
- Bias of parameter i in CR units: `mean_r(theta_hat_r,i - theta*_i) / sqrt((J^-1)_ii)` over the 500 Part B experiments.
  "Largest bias" is the largest absolute value over the nine parameters.
- `D-hat = -log det M-hat`, `M-hat = (1/500) sum_r e_r e_r^T`, `e_r = theta_hat_r - theta*` (errors about the true theta,
  bias included). "D-hat more than 10 nat below I" means `I - D-hat > 10`; "within 0.3 nat of I" means `|D-hat - I| <= 0.3`.
- A CML run is "not accepted" iff neither of its optimiser calls is accepted by the round-8b rule
  (`estimator_r8b.accepted`), i.e. the run falls back to `theta_LS` (convention of round 8b `estimate_stage1`).
  "At most 2 %" means at most 10 of 500 runs, for each trigger separately.
- Part C differences: the point estimate is the mean over the ten repetitions (noise seeds 20261117 + b, b = 0..9, lambda
  fixed at the tuned value) of the paired difference of `I`; each repetition's `I` is `log det` of the mean `J_cond` over its
  2000 trajectories.

**Lines.**
1. **A1.** Hit iff sd(RULE) in [0.04, 0.09], sd(SOD) in [0.20, 0.33], I − mean (RULE) <= 0.005 and I − mean (SOD) in [0.02, 0.05].
2. **A2.** Hit iff I − log det(mean J_sent) is in [12.8, 13.8] for RULE, in [3.3, 4.1] for SOD and in [14.0, 15.5] for PRED.
3. **B1.** Hit iff, for LS under RULE, the largest bias > 3 and I − D-hat > 10.
4. **B2.** Hit iff, for LS under SOD, the largest bias > 0.5.
5. **B3.** Hit iff, for CML under RULE and under PRED: largest bias < 0.2, |D-hat − I| <= 0.3, and at most 10 runs not accepted.
6. **C.** Hit iff I(C1) − I(C0) in [0.00, 0.20] and I(C2) − I(C1) in [−0.20, +0.05].

## Estimator procedure (fixed before running; DECISION 18-6)

- **LS** is `estimator_r8.ls_init` on the transmitted pairs (numpy `lstsq`; rank and finiteness recorded).
- **CML** is `estimator_r8b.estimate_stage1(data, restart_seed)` unchanged: first L-BFGS-B call (`estimator_r8.run_lbfgsb`)
  from `theta_LS`, restart call from `theta_LS + 0.01 s * d(restart_seed)`, the round-8b Newton-decrement acceptance,
  fallback `theta_LS`. Restart seed `80261116 + i` for experiment i. `data` carries the transmitted pairs and the windows
  `(lo_k, hi_k)` of the non-transmitted steps, reconstructed from the record (the last transmitted value `r_k`, by
  `estimator_r8.references`) and, for `RULE` and `PRED`, the design output `x_k`. For `SOD` the object is
  `estimator_r8.StageData(ev, z, ('sod',))` itself.
- Statistics use the decided theta of every run (fallback included), as round 8b. If any run falls back, the statistics over
  the accepted runs only are reported as well.
