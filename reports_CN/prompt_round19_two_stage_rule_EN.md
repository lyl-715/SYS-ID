# Round 19: complete two-stage experiment with the one-parameter rule

## 0. Rules (unchanged)
- Frozen branches are read-only: claude/optimistic-newton-xv5jvl @3314e7d, round10-upper-bound @2b1af1b, round11-edge-actions @1b88f01.
- This round uses a new branch round19-two-stage-rule and a new folder round19/. If number 19 is already taken, use the next free number and say so in the README.
- Branch off from the most recent branch that contains the one-parameter rule and the count-feedback code. State the starting branch and commit in the first line of the README.
- Predictions are committed before any code that produces results is run: round19/predictions.md must be a separate commit that precedes all results.
- No force-push.
- If this prompt contradicts the existing code, the code wins. List every such case in a "Deviations" section of the README, with the reason.
- No tuning after seeing results, and no variants beyond those listed here. If something fails, report it; do not repair it by redesigning.

## 1. Purpose
The current +10.30 nat holds with a known model. This round answers: if the model is not known in advance, is first estimated with constant send-on-delta (SoD), and the one-parameter rule is used afterwards, how much information does the whole experiment gain over constant SoD throughout, and is the budget kept?

## 2. Scenario (unchanged)
G*(s) = 2/(1+s)^2, frequencies (3*sqrt(2), 3*pi, 12, 12.3) rad/s, sampling interval 1 ms, K = 5000, p = 9,
Delta* = 0.002384694482635785, sigma = Delta*/20, B = 316.8921168493721.
Baseline: constant SoD throughout, logdet J_SoD = 197.20097668224076. Full transmission: 219.00850592919505.

## 3. Two-stage procedure
Stage 1 (steps 1..K1): constant SoD with threshold Delta*.
End of stage 1: the receiver computes the censored maximum-likelihood estimate theta_tilde from the stage-1 record (transmitted values plus the interval at silent steps).
  Use the existing censored-ML implementation and do not change its acceptance rule. Replications in which it does not converge are counted and listed separately; they are not dropped silently.
Stage 2 (steps K1+1..K): one-parameter rule
  lower edge = min(xbar_k - e_k*sigma, r_k - sigma), upper edge = max(xbar_k + e_k*sigma, r_k + sigma), xbar_k = h_k^T theta_tilde.
  r_k carries over from stage 1 (the last value transmitted in stage 1).
  e_k follows the existing count feedback (the form of eq. (fb) in the reply note, not the "improved" variant):
    e_k = e_0 + (N_k - plan_k) / (2 * q_2 * T), clipped to [0.5, 4],
    e_0 = 1.577, T = 100,
    q_2 = (B - 3 - N_1) / (K - K1), plan_k = N_1 + q_2 * (k - K1),
    where N_1 is the actual number of transmissions in stage 1 and N_k the total number of transmissions before step k.
  No fallback mechanism.
Final estimate: censored maximum likelihood on the whole record (both stages), with the window at every step given as the receiver can reconstruct it.

## 4. Configurations (all fixed in advance)
Primary: K1 = 2500, 1000 replications.
Secondary: K1 = 1250 and K1 = 3750, 300 replications each.
Control (K1 = 2500 only, 300 replications): stage 2 uses the true model theta* (everything else identical), to separate the cost of the first stage from the cost of model error.
Baseline: constant SoD throughout, paired with the primary configuration through the same noise sequences.
Seeds are fixed and written in the README. If the run time is excessive, the primary configuration may be reduced to 300 replications; say so.

## 5. Two checks first (if either fails, stop and report; do not continue)
(a) Run "constant SoD throughout" with this round's simulation and information computation: logdet must be within 3 standard errors of 197.201, and the expected number of transmissions within 3 standard errors of 316.89.
(b) Run "rule throughout, true model, no feedback, e = 1.577": logdet must be within 3 standard errors of 197.201 + 10.30.

## 6. Quantities to report
Information: J = sigma^-2 * sum_k wbar_k h_k h_k^T, where wbar_k is the average over replications of c(alpha_k, beta_k) at step k (computed with the true x_k and the realised window).
  Report the gain in logdet J over constant SoD throughout, with a 95% interval from a bootstrap over replications.
  If the earlier two-stage experiment with the old table (3 Nov draft, net gain 3.67 nat) used a different measure, report both and state which one the earlier experiment used.
Budget: mean, standard deviation and maximum of the total number of transmissions; fraction of replications above B; transmissions in stage 1 and stage 2 separately.
Model: distribution of the prediction error of theta_tilde (root mean square of predicted minus true output, in units of sigma).
Estimator: bias of the final estimate (largest over the 9 parameters, in Cramér–Rao standard deviations); ratio of the empirical mean-square error to J^-1 (expressed in logdet).
  The known shortfall of about 3% of ML relative to the CR bound under the rule is only to be reported, not fixed.
Failure cases: number of stage-1 ML non-convergences; number of replications in which e_k hits 0.5 or 4; number of replications with an abnormal stage-2 count (fewer than 10 transmissions, or more than twice the plan).

## 7. Predictions (copy verbatim into round19/predictions.md and commit first)
These are the predictions of the prompt author. You may add your own in a separate section of the same file, labelled as yours.
- K1 = 2500: net gain 6.0 nat (range 5.3–6.6). Basis: if the information matrices of the two stages are roughly proportional, 9*ln(0.5*(1+exp(10.30/9))) = 6.55, minus the losses from model error and feedback.
- K1 = 1250: net gain 8.1 nat (range 7.3–8.6).
- K1 = 3750: net gain 3.7 nat (range 3.2–4.0).
- Control (true model in stage 2, K1 = 2500): 6.4 nat.
- Total transmissions: mean between 312 and 316; about 10% of replications above B (range 3%–25%).
- The budget is most at risk for K1 = 1250: its fraction above B is higher than for K1 = 2500.
- Final estimate: largest bias below 0.3 CR standard deviations; mean-square error at most 5% above the CR bound.

## 8. Deliverables
round19/README.md: starting commit, seeds, results of the two checks, results table (primary, two secondary, control, baseline), predictions versus results item by item, a "Deviations" section, a "Not done" section.
Raw results saved to file (one row per replication: N_1, N_2, prediction error, convergence flag, etc.).
No figures. Push the new branch; do not open a pull request.
