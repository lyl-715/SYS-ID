# Decisions — round 2d (overnight mode)

Spec: `reports_CN/plan_round2d_CN.md` (frozen, commit ac058c4). Predictions: `predictions_round2d.md` (commit a2c4050).
Where the spec or the task leaves a choice, the reading taken is logged here with the alternatives considered. No decision
below tunes λ, Ã, the trust-region rule or the stopping rule to a target, and no number is called a bound; everything
produced is an estimate on the discrete model (估算).

## Part A (objective and oracle)

- DECISION A-1 (square root of M_0). The spec writes A = M_0^{-1/2} Ã M_0^{-1/2}; the task asks for g_k via Cholesky with no
  explicit inverse. Taken: M_0 = L Lᵀ (lower Cholesky factor, round 2c `factor()`), A = L^{-T} Ã L^{-1},
  g_k = σ⁻¹ L⁻¹ h_{U,k} (one triangular solve). L⁻¹ = Q M_0^{-1/2} with Q = L⁻¹ M_0^{1/2} orthogonal, so this is the spec's
  parametrisation after the orthogonal change of variables Ã ↦ Qᵀ Ã Q: D, log det Ã, the Frobenius trust region and
  ‖Ã − I‖_F are unchanged, and Ã = I is round 2c's A_1 in both. Alternative: form M_0^{-1/2} by an eigendecomposition
  (not taken: the task asks for Cholesky; the results would differ only by rounding).
- DECISION A-2 (bitwise reuse of round 2c at Ã = I). γ_k(Ã) = einsum(VW, Ã·VW)/σ² with VW = L⁻¹ H_Uᵀ; Ã·VW is put in VW's
  (Fortran) memory layout, so at Ã = I the arithmetic is round 2c's `gamma_of` bit for bit (checked in B1). The constant
  log det M_0 − log det Ã − 9 + γ_1 is summed in round 2c's order, and D = const + Ṽ + 315.892 λ as in round 2c.
- DECISION A-3 (oracle = DP with stored policy + round 2c forward). The DP is round 2c's `dp2d_gamma` loop (same kernels
  `_q_all`, `_wmax`, same order); in each step the A = 1 policy is also stored with round 2b's `_codes` (round 2a tie rule:
  smallest |v| first, then +v before −v) and `_pack`, in RAM (4999 × 247569 bytes = 1.15 GiB, reused every call; no disk
  memmap). `_codes` returns the number of states where the stored action's Q + r differs from the window maximum; it is
  logged every call (must be 0). The forward pass is round 2c's `forward2d_steps` unchanged, fed by `_unpack`.
  w_k = E_π[c(δ_k)] for k ≥ 2, w_1 = 1, n = Σ_{k≥2} E_π[q(δ_k)]. Diagnostic every call (no pass line):
  Σ_{k≥2} γ_k w_k − λ n against Ṽ (forward–backward consistency of the stored policy).
- DECISION A-4 (subgradient). G_Ã = −Ã⁻¹ + g_1 g_1ᵀ + M̃^{(2+)}, with Ã⁻¹ from the Cholesky factor of Ã (`cho_solve`), and
  G_λ = 315.892 − n, exactly as the task writes them. When the DP has ties, the stored policy is one maximiser; any
  maximiser gives a valid subgradient of the max-of-linear Ṽ (spec §1.2), so no tie handling beyond the round-2a rule.
- DECISION A-5 (B − 1). 315.892 as written (`grid30.LAM_BUDGET`, as rounds 2a–2c; round 2c DECISION C-1).

## Part B (checks)

- DECISION B-1 (random directions). Δ = (Z + Zᵀ)/2 / ‖(Z + Zᵀ)/2‖_F with Z a 9 × 9 standard normal matrix from
  `numpy.random.default_rng(20261022)` (seed = the spec's date, fixed before the run; three draws in order). Alternative:
  any other seed (no reason to prefer one; the seed is recorded so the directions can be regenerated).
- DECISION B-2 (how D is evaluated off the base point). D(I + εΔ, λ) and D(I, λ ± ε) use round 2c's `dp2d_gamma` (no policy
  storage; it returns the same Ṽ bit for bit as the storing DP, checked in development at λ*_2c); only the base point
  (I, λ*_2c) needs the policy and the forward pass for G. The slack of each test is also split into the log-det part
  −log det(I + εΔ) + ε tr Δ (≥ 0, ≈ ε²/2) and the rest (γ_1 + Ṽ part, ≥ 0 by convexity of Ṽ), both logged; the pass line
  is the task's total slack ≥ −1e-9.
- DECISION B-3 (gate). `logdet_outer.py outer` refuses to run unless `results/checks.json` records B1 and B2 passed. The
  spec's checks 3 and 4 (cut validity, m_global non-decreasing) are computed during the run (Part C, C4). Edits to
  `logdet_outer.py` after the checks ran (11:09) touched only the master problem (SCS tolerance, positive-definiteness
  test of the returned Ã) and the final/table/plot stages; the oracle and D paths used by B1/B2 are unchanged.

## Part C (outer loop)

- DECISION C-1 (iteration bookkeeping). Iteration j = one oracle call at (Ã_j, λ_j), j = 0..299 (300 oracle calls at
  most). In iteration j: oracle; trust-region ratio for the point just evaluated, ratio = (D_c − D_j)/(D_c − m_j) with the
  centre value D_c and the model value m_j predicted for this point in iteration j − 1; accept (ratio ≥ 0.25: centre ← x_j,
  r ← min(1.5 r, 2)) or reject (centre stays, r ← max(r/2, 1e-3)); add cut j (always); solve the master problem without
  the trust region (m_global, gap = D_best − m_global, stop if ≤ 0.02); otherwise solve it with the trust region around
  the centre (next point x_{j+1}, m_{j+1}); cut-validity check over all evaluated points; checkpoint. If the predicted
  decrease D_c − m_j is ≤ 0, the ratio is undefined; it is logged as −inf and the step counts as rejected. The spec's
  rule keeps the centre on a rejected step even when that step's D is below D_c; followed as written (D_best still takes
  every evaluated point).
- DECISION C-2 (solvers). cvxpy 1.9.3; SCS 3.3.1 first with eps_abs = eps_rel = 1e-8, max_iters = 100000 (SCS's defaults
  1e-4 are coarse next to the 0.02-nat stopping test and the m_global monotonicity report); if SCS's status is not
  'optimal' (e.g. 'optimal_inaccurate') or its Ã is not positive definite, the same problem is solved with Clarabel 0.11.1
  at its defaults. The tolerance was set in development on synthetic cuts (no oracle data, before the outer run): at 1e-9
  SCS ended 'optimal_inaccurate' after ~6 s per solve; at 1e-8 it returned 'optimal' in < 1 s with a value within 1e-8
  of Clarabel's. This is a solver setting, chosen for status and speed, not for any reported number. Every master
  solve logs the solver used, its status and, on fallback, SCS's status. The constant log det M_0 − 9 is left out of the
  cvxpy objective and added after the solve (keeps the solver's relative tolerances on the O(10) part). Ã is a symmetric
  PSD variable; −log_det keeps it in the interior.
- DECISION C-3 (which model value is used where). m_global = the solver's optimal value of the master problem without
  the trust region (plus the constant), as the spec defines it. m_{j+1} (the prediction in the ratio) = the analytic
  cutting-plane model evaluated at the returned next point, so the ratio compares D and the model at the same point;
  the solver's value is logged next to it (the difference is the solver's accuracy). The returned Ã is symmetrised
  ((X + Xᵀ)/2) and λ clipped to [0, 0.5] (clip amount logged).
- DECISION C-4 (trust region on Ã only). As the spec writes it, ‖Ã − Ã_c‖_F ≤ r constrains Ã only; λ is limited by
  [0, 0.5] alone, so the λ coordinate moves like plain Kelley cutting planes and a rejected step caused by λ also halves
  r. Followed as written; alternative (a trust region on λ too) not taken.
- DECISION C-5 (cut validity, spec §3 item 3). After each master solve: for every evaluated point i,
  model(x_i) − D_i = max_l cut_l(x_i) − Ṽ_i ≥ 0 trivially (l = i gives Ṽ_i). The reported "largest violation" is
  max_{i, l ≠ i} [cut_l(x_i) − Ṽ_i]: > 0 would mean D_i < model(x_i), i.e. a cut that cuts off an evaluated point. Its
  self term |cut_i(x_i) − Ṽ_i| (rounding only) is logged next to it.
- DECISION C-6 (checkpoint). After every iteration, `results/cuts.npz` (all cuts: Ã_i, λ_i, Ṽ_i, D_i, M̃_i^{(2+)}, n_i, …,
  the history columns, the centre and the next point) and `results/state.json` (centre, r, D_c, next λ, m_next, solver
  records) are written to temporary files and moved into place (atomic replace). A restarted `outer` resumes at the
  stored next point. Per-iteration w_i and E[q] per step go to `results/w_iter/` (not committed: 80 kB per iteration);
  w* of the best point is committed as `results/w_star.npy`.
- DECISION C-7 (final point). (Ã*, λ*) = the evaluated point with the smallest D (D_best), not the last centre; n* and
  w* are that point's own policy's forward values.

## Part D and report

- DECISION D-1 (the +0.701 row). The real evaluated optimum is NM-none-P5514's Δm_D = 0.7012655152921639
  (`round2a/results/recheck7.json`, evaluator_fast B, n = 24), relative to the evaluator's log det J(u_unif) =
  197.20097668224076, not to 197.197; on the −197.197 scale of the other rows it is 197.90224219753293 − 197.197 =
  0.70524 (logged next to it; the 0.0044 offset is round 2c DECISION A-5). The row is shown as the task gives it (+0.701),
  with the offset stated.
