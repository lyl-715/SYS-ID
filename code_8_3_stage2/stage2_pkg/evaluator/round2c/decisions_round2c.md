# Decisions — round 2c (overnight mode)

Spec: `reports_CN/plan_round2c_CN.md` (frozen, commit a716952). Predictions: `predictions_round2c.md` (commit cae8325).
Where the spec or the task leaves a choice, the reading taken is logged here with the alternatives considered. No decision
below tunes λ or A to a target, changes the frozen ε_λ = 4e-6, or calls any number a bound; everything produced is an
estimate on the discrete model (估算).

## Part A (construction)

- DECISION A-1 (source of the weights). The task says w(û) and w_U are archived in `round2a/results`; that folder holds
  J(u) only (`round2a/recheck7.py` saves `J_*.npy`, not w). w_U is archived at `evaluator/results/fast/w_B_U_n24_m8.npy`
  (speed-up regression archive); w(P5514) is archived nowhere. All eight weight vectors (U and the seven candidates, the
  latter needed by B2) were therefore regenerated with the same deterministic call as `round2a/recheck7.py`
  (`evaluator_fast.evaluate_fast(name, 24)`, B, n = 24, margin 8Δ; candidates added to the evaluator's in-memory input
  table only). Acceptance, all passed (`logs/part_a.log`, A2 lines): J bitwise equal to `round2a/results/J_<id>.npy`, S,
  E[N] and log det J bitwise equal to `round2a/results/recheck7.json`, zero window-assertion failures, and the recomputed
  w_U bitwise equal to `results/fast/w_B_U_n24_m8.npy`. Archived as `round2c/results/w_<id>.npy`. Alternative: stop and
  ask (not taken: overnight mode; the evaluator is deterministic and the bitwise checks identify the weights).
- DECISION A-2 (h_U). h_{U,k} = `density_recursion_v1.H`, i.e. mc83.py's H (a_m = 1/2, χ_m = 0, a0 = 1) exec'd by the
  evaluator's setup block; it equals `regressors(1/2, 0, t)` and the evaluator's U regressors bitwise (A1 line 1).
- DECISION A-3 (orientation of R). The spec writes "R 为 2×2 旋转" without a sign. Taken: R(χ) = [[cos χ, sin χ],
  [−sin χ, cos χ]], the rotation that maps the uniform block (½ cos ω_m t, −½ sin ω_m t) to the block
  (a_m cos(ω_m t + χ_m), −a_m sin(ω_m t + χ_m)) of mc83.py's regressor after the factor 2a_m. Verified numerically for
  P5514 and U-L1657 (max |h_k(u) − T(u) h_{U,k}| = 3.5e-15 and 2.8e-15).
- DECISION A-4 (numerics of M, A, γ). M(w) is assembled with the evaluator's own J expression
  `(H*w[:, None]).T @ H / σ²`, so M(w_U) equals the archived J(u_unif) bitwise (T(u_unif) = I; logged). One Cholesky
  factor L per A: log det A_i = −2 Σ log diag L; A_i = `cho_solve(L, I)` (two triangular solves, stored for B2 and the
  record); γ_k = σ⁻² |L⁻¹ h_{U,k}|² by a triangular solve (the stored A_i reproduces γ to 1.1e-15 relative). Condition
  numbers are 2-norm (`numpy.linalg.cond`). Alternative: γ_k = h_kᵀ A h_k with the stored A (logged as a diagnostic only).
- DECISION A-5 (log det J_C). The spec's 197.197 is used as written (= `bound/bound_common.py` LD_SOD
  = 197.19655785394153 rounded: the R = 10000 Monte-Carlo weights of the uniform input; the same constant as
  `round2a/b5_conversion.py` LD_REF). The evaluator's log det J(u_unif) = log det M_2 = 197.20097668224076 differs by
  0.0044; it is not substituted.
