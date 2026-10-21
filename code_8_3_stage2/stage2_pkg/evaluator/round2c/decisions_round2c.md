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

## Part B (checks)

- DECISION B-1 (when an inequality "holds" in B2). Slack ≥ −1e-10 nat counts as holding (log det ≈ 197, so float64
  rounding of a log det is ~1e-13). Needed for P5014 (equal powers p_m = 1/4, power term exactly 0): slack1 = −1.7e-13,
  the rounding difference of two slogdet calls on matrices related by the rotation T. The P5514/A₁ equality test is
  |slack2| ≤ 1e-8 as specified. Alternative: slack ≥ 0 strictly (would fail P5014 on rounding alone).
- DECISION B-2 (how the terms of B2 are computed). tr(A_i M(w(u))) = Σ_k w_k γ_k^(i), the per-step form the DP uses;
  tr(cho_solve(L_i, M(w(u)))) is logged next to it as a diagnostic. log det J(u) and log det M(w(u)) by `slogdet` (as
  the evaluator); log det A_i = −log det M_i from the Cholesky factor of Part A. slack1 is also compared with its
  closed form −2 Σ_m log(4 p_m) (logged).
- DECISION B-3 (B4 quantity). The compared number is Σ_{k=2..K} γ_k E[c(δ_k)] (the γ₁·1 term of the first, forced sample
  is a constant and is logged separately), for γ^(1) and γ^(2). Forward: E[c(δ_k)] per step from the round-2b 2-D
  transport (`m2._fwd`), summed with γ_k; backward: round 2b's fixed-policy backward with the reward γ_k c. The open-loop
  policy is round 2b's 3.3 policy (`m2.open_loop_codes`).
- DECISION B-4 (extras, not spec checks). Same tolerance 1e-10, logged as "extra": the 1-D DP with γ ≡ 1 at round 2b's
  1-D λ* = 1.9185615985194624 against `round2b/logs/gold_1d.jsonl:17`, and the 1-D forward–backward of B4. They cover
  the 1-D driver used in C1, which B3/B4 as specified test only in 2-D.
- DECISION B-5 (gate). `logdet_dual.py outer` refuses to run unless `results/checks.json` records all four checks passed.
- DECISION B-6 (Part B run twice). The first run (logs/part_b.log, first block) printed four B4 numbers as
  `np.float64(...)`; the cast was fixed and Part B re-run unchanged otherwise. Both blocks stay in the log; the numbers
  are identical; results/checks.json is from the second run.

## Part C (estimates)

- DECISION C-1 (the constant B − 1). 315.892 exactly as the task and spec write it (`grid30.LAM_BUDGET`, as rounds 2a/2b,
  round2b DECISION A-1), not (B_full − 1) = 315.8921168…; the difference is 1.17e-4·λ, below 1e-4 nat for λ ≤ 0.44.
- DECISION C-2 (outer loop). Round 2b's golden section (`grid30.golden`, same arithmetic) on [0, 20 γ̄_i] with γ̄_i the
  mean of the γ^(i) in use (20 γ̄_1 = 0.38565362149452176, 20 γ̄_2 = 0.4317792308978602), stopped when the bracket width
  ≤ ε_λ = 4e-6; λ* = the evaluated λ with the smallest D; no refinement (spec §2). Each Ṽ is archived per λ in
  `logs/gold_<model>_A<i>.jsonl` (key repr(λ)) and in `results/outer_<model>_A<i>.json`. The constant of D is
  −log det A_i − 9 + γ_1^(i) = log det M_i (Cholesky, Part A) − 9 + γ^(i)[0].
- DECISION C-3 (scheduling). The two 2-D outer loops ran one after the other with 4 numba threads; then the two 1-D
  loops ran concurrently, one process each (the 1-D kernels are single-threaded). Every per-λ time carries the thread
  count and the load average.
- DECISION C-4 (log files). One log per case, `logs/part_c1_<model>_A<i>.log`. The first launch wrote to
  `logs/part_c1.log`; to keep the concurrent 1-D runs from interleaving in one file, that file was renamed to
  `logs/part_c1_2d_A1.log` while the 2-D A₁ run continued (same process, not restarted; its open handle kept writing
  there) and the file name was made per case for the later runs. No completed λ evaluation was recomputed (see C-7).
- DECISION C-5 (reading of L7). "二维外层总耗时" is read per outer loop (spec §2 sizes one loop: ~25 evaluations × ~1 min),
  as the sum of the per-λ DP times of that loop; L7 is judged for each A, and the sum over both A is reported as well.
- DECISION C-6 (S-route value). U(619.1292140245644; u_unif) = the route-1 certificate value U_cp of
  `round2a/b5_conversion.solve` (exchange with cap 1, tol 1e-10, batch 256, then certificate_full), at the full-precision
  S̃_acc of round 2b; its solver gap is logged. The solver is first re-run at W = 1210.9563921988488 and compared with
  `round2a/logs/b5_conversion.log` line 6 (reproduction check, no pass line).
- DECISION C-7 (interruption of the 2-D A₂ loop; restart from the λ cache). The 2-D A₂ loop was started at 09:34:38 by a
  waiting shell script launched with plain `nohup … &`. At about 09:40 the process died together with that shell when
  a foreground tool call of the session was interrupted (no Traceback; last log line 09:39, the 13th evaluation). The 13
  completed evaluations were intact in `logs/gold_2d_A2.jsonl` (all lines parse). The loop was restarted at 09:46:46
  with `setsid nohup sh run_c_rest.sh` (own session, survives the session's tool calls), which then runs the two 1-D
  loops, C2 and C3. The golden section is deterministic, so the restart replayed exactly the 13 archived λ in the
  original order (`logs/part_c1_2d_A2.log`, second block, lines marked "cached from logs/gold_2d_A2.jsonl"):
  0.1649249905666938, 0.2668542403311664, 0.10192924976447254, 0.06299574080222124, 0.038933508962251284,
  0.024062231839969944, 0.01487127712228134, 0.029742554244562687, 0.02055159952687409, 0.01838190943537719,
  0.021892541748473045, 0.022721289618370992, 0.021380347396772036; the 14th λ, interrupted before it was archived, and
  all later λ were computed after the restart. The first block of that log (09:34:38–09:39) stays as written. L7 uses
  the sum of the per-λ DP times recorded in the cache (each λ once), not the wall clock, which includes the gap.
- DECISION C-8 (reading of L4). L4 names no model; L2 and L3 are about the 2-D model and L5 is the only prediction
  stated for the 1-D model, so L4 is judged on the 2-D model (A₂ − A₁ = 0.875 nat). The 1-D difference (2.201 nat) is
  reported next to it and lies outside 0.2–1.0.
- DECISION C-9 (figures). `fig_gamma.png` (γ_k of both A over t_k) and `fig_dual_lambda.png` (D − 197.197 at the
  evaluated λ of each golden section, one panel per model, own y scale, x shown up to 5 γ̄ of the searched 20 γ̄, the
  S-route value as a dashed line) are drawn by `plots.py` from `results/`; no reported number is read off a figure.
