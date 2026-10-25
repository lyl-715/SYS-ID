# Decisions — round 8 (spec `reports_CN/plan_round8_CN.md` v3.1, frozen)

Each line records a choice the spec (and the task) does not make, or a point where the frozen text cannot be executed as
written, the reading taken and where it acts. Overnight mode: none of these stopped the run. Nothing was tuned to a target
using the true error. Wording: 本场景、固定设计规则下的序贯仿真结果.

## P — before the predictions commit

- DECISION P-1: before the predictions commit (d73f995) a scratch prototype (not committed) ran scipy L-BFGS-B with the frozen
  options on eight stage-1 SoD records generated with seeds 20261029–20261036, which are the stage-1 noise seeds 20261028 + r of
  r = 1..8. It looked only at the optimiser's termination (message, gradient infinity norm, iterations) and compared two
  arithmetic forms of the same objective. No ε_1, estimation error, transmission count or design quantity was computed. The
  predictions are the spec's Q1–Q7 verbatim, so they could not be influenced. Every later engineering probe used seeds
  ≥ 91000000 (disjoint from all pipeline seeds).

## E — the estimator

- DECISION E-1 (the frozen convergence threshold was not met in any of the 3000 calls of the frozen implementation; wording
  corrected in round 8b Part 0, which replaced "cannot be reached in double precision" -- see report_round8_CN.md §0 item 1
  for the reviewer's extra Newton-step diagnostic, outside the frozen algorithm). Spec §1.2 accepts convergence only
  when ‖∇_ϑ ℓ‖∞ ≤ 1e-8 for the SUMMED negative log-likelihood. Here the Hessian in ϑ has eigenvalues from 3.5e8 to 1.35e11
  (`logs/partA_checks.log`, A3-1e; σ = Δ*/20 = 1.19e-4, so each transmitted sample contributes 1/σ² = 7e7). Changing θ_1 by
  one unit in the last place changes the raw gradient by 1.4e-5, which is 1400 times the threshold (A3-1e). L-BFGS-B with
  ftol = 0 stops when the objective value stops changing in floating point. In centred arithmetic (E-2) that happens at
  ‖∇_ϑ ℓ‖∞ ≈ 1e-4 to 1e-1, with the Newton decrement at 1e-9 to 1e-6 standard errors (probe seeds 91000000–91000039, 40
  records: first-call decrement ≤ 3.0e-7, restart ≤ 1.7e-6). So under the frozen rule every estimation is "not converged" and
  goes to its fallback. Reading taken (both are reported; only the first counts):
  - **Primary analysis = the frozen rule exactly as written** (acceptance requires ‖∇_ϑ ℓ‖∞ ≤ 1e-8). The verdict of spec §5 and
    the scoring of Q1–Q7 use only this analysis.
  - **Supplementary analysis ("补充分析 S", NOT the frozen rule):** identical pipeline (same noise, same optimiser runs, same
    restart rule, same fallbacks, same design rule). The only change is the last acceptance condition: ‖∇_ϑ ℓ‖∞ ≤ 1e-8 is
    replaced by the Newton decrement in ϑ, sqrt(gᵀ H_ϑ⁻¹ g) ≤ 1e-5 (H_ϑ = analytic observed Hessian in ϑ, Cholesky; a
    Cholesky failure means not accepted). This is a remaining optimisation error below 1e-5 statistical standard errors. The
    threshold was set on the probe seeds above (margin ≥ 30 over the first-call maxima) before any pipeline run, and never
    with the true error. Because θ̂_1 differs between the two rules, the supplementary analysis has its own designs, tables,
    stage-2 records, post-hoc evaluations and joint estimates. The control's stage-2 record is shared, since it does not
    depend on θ̂_1. The report gives the §5 readings of S as "按 §5 规则的读数（非冻结规则，不作判定）".
- DECISION E-2 (centred arithmetic): each optimiser call evaluates the objective about its own start point θ_ref. Residuals
  (z_k − h_kᵀθ_ref)/σ and window ends (lo_k − h_kᵀθ_ref)/σ are formed once, and θ = θ_ref + s ⊙ v with v = ϑ − ϑ_ref the
  L-BFGS-B variable, started at v = 0. It is the same objective and the same optimiser (L-BFGS-B is translation invariant
  without bounds); only the floating-point representation of the parameter offset is finer. Without it the gradient's
  rounding floor alone is about 1e-5 (A3-1e). The reported θ̂ = θ_ref + s ⊙ v̂.
- DECISION E-3: the θ-independent constants (log σ + ½ log 2π per transmitted sample) are dropped from the summed objective,
  as in mc84.py. This changes neither the gradient nor the minimiser.
- DECISION E-4: for every estimation both optimiser calls are run: the first from θ_init (stage 1) or θ̂_1 (joint), and the
  restart from that start point + δ. Under each rule the restart's result is used only if the first call is not accepted.
  Running the restart when it is not needed is deterministic and does not change any rule's outcome. Both calls' diagnostics
  are archived.
- DECISION E-5: "rank 9" for LS usability is `numpy.linalg.matrix_rank` with its default tolerance. LS = `numpy.linalg.lstsq`
  (rcond=None) on the transmitted samples of stage 1.
- DECISION E-6: L-BFGS-B options other than gtol = 1e-8, ftol = 0, maxiter = 500 are scipy's defaults (maxcor 10, maxls 20,
  maxfun 15000).
- DECISION E-7: the stage-2 window of a non-transmitted step k under table T is (r_k − M⁻ h, r_k + M⁺ h). Here r_k is
  reconstructed from the record, (M⁻, M⁺) is looked up from r_k (tables_r8.SeqTable, the same rule as the simulator), and
  h = σ/24. For an SoD stage the window is (r_k − Δ*, r_k + Δ*). The likelihood is mc84.py's (same log P forms, same gradient
  and Hessian expressions) with these per-step ends in place of r_k ∓ Δ*. Check A3-1d confirms it equals mc84.py's
  nll_grad_hess on an SoD record (relative ≤ 1e-14).

## A — design module, simulator, post-hoc evaluation (Part A)

- DECISION A-1 (reuse without importing θ_0): the frozen modules the design must reuse (density_recursion_v1, evaluator_fast,
  round4/evaluator_window, round4/trigger_dp) all execute the mc83.py setup block at import, which builds the true frequency
  response and the nominal output. `build_kernels.py` therefore copies the needed top-level functions, classes and constant
  assignments verbatim (exact source lines found with `ast`) into four generated modules r8k_dr / r8k_ef / r8k_ew / r8k_td.
  These mirror the originals' structure (references like `dr._kernel` resolve to the mirror). Each has a short new header:
  imports, constants from declared.py, and in r8k_dr `input_arrays(name)` returning the (x̂, H) pair the design registered.
  checks_r8.py re-extracts every block and requires byte equality (57/57), checks that the functions use no undefined global
  names, and the design-chain regression reproduces T4 bit for bit through the mirrors.
- DECISION A-2: x̂_k = h_kᵀθ̂_1 is evaluated by `declared.x_polar`, the polar form of the declared model structure:
  θ_1 a0 + Σ_m a_m |g_m| cos(w_m t + χ_m + arg g_m) with g_m = θ_{2m} + iθ_{2m+1}. This is the arithmetic of mc85_scan.output,
  so θ̂ = θ_0 gives the round-4 nominal output bit for bit. H θ_0 in another summation order differs by up to ~1e-12 σ
  (R-DECL H_theta0_close), which would break bit-for-bit T4.
- DECISION A-3: λ = 4γ̄ is formed as 0.8·(5·γ̄), the floating-point arithmetic of the round-6 scan run that defined T4
  (round7/tables.py). At the nominal γ̄, 4.0·γ̄ differs from it in the last bit.
- DECISION A-4 (the allowed design failures of spec §1.7, exactly): (i) J_SoD_hat non-finite; (ii) Cholesky failure of
  J_SoD_hat (scipy LinAlgError, in a pre-check with cho_factor or inside round 4's Design.gamma); (iii) A non-finite (A is formed
  with cho_solve only for this check and for the archive; γ_k is round 4's verbatim Cholesky solve); (iv) γ_k, γ̄ or λ
  non-finite; (v) nominal output non-finite; (vi) an ArithmeticError raised in the DP; (vii) "no selectable action", i.e. a
  node-step whose Bellman maximum is not finite (round 4's `nbad` > 0); (viii) a non-finite DP value. Everything else
  propagates and stops the run, in particular AssertionStop and AssertionError from the evaluators' conservation /
  non-negativity / support assertions, the grid assertions in Design, and the sentinel's SentinelAccess.
- DECISION A-5 (table lookup, simulation): the cell of the actual reference r on the table's own coarse grid is
  i = floor((r − z0)/h + 1/2), as in round 7's cell_index. There is no boundary assertion for real-valued r (a tie has
  probability 0; the floor rule takes the upper cell). A cell outside the step's stored range takes the range's boundary node
  (rounds 5/7). Transmit iff (z_k − r) ≥ M⁺h or (r − z_k) ≥ M⁻h. For an all-(480, 480) table this equals mc83.sod's
  |z_k − r| ≥ Δ* exactly, because 480h = Δ* in floating point and fl(r − z) = −fl(z − r) (R-SIM: 100/100 trajectories).
- DECISION A-6 (post-hoc grid): each table carries its own coarse origin z0 from x̂, so the σ/120 quadrature grid is anchored on
  that table's lattice: z_j = z0 + (n0 + j)h_120, with n0 = round((min x0 − 8Δ − z0)/h_120). G is the node count of round 7's
  grid of the true output. For x̂ = x0 this is round 7's grid bit for bit (n0 = 0; R-POST reproduces round 7's T4 and SoD σ/120
  records bit for bit). For odd f no node lies on a cell boundary (the smallest gap is 0.1 cell widths). Support windows are
  round 4's R = 88σ centred at x0.
- DECISION A-7 (packed tables): bz2 (level 9) of an uncompressed npz holding the two uint8 planes i₋ = code // 20 and
  i₊ = code % 20 (pad 255), start, width, z0, h, G, λ, γ̄. The round trip is checked on every write (T4: 42 MB raw → 69 KB).
  One file per replication and analysis in `results/tables_packed/`; the codes' sha256 goes into the archive.
- DECISION A-8 (the four correctness checks, tolerances not fixed by the spec):
  - A3-1(a): pass iff the estimator's returned value satisfies max_i |θ̂_i − θ̂^LS_i| / s_i ≤ 1e-8.
  - A3-1(b): the archived §8.4 ML result is `stage2/final_theta_mle.npz` 'UNIF_mle' row 0 (uniform input, seed 20261014,
    mc84.py's damped Newton, stopping at decrement ≤ 1e-9 max(1, |f|), i.e. ≤ ~6e-4 standard errors). The record's identity
    is confirmed by LS on its sent samples equal to the archived 'UNIF_ls' row 0. Pass iff
    sqrt(Δθᵀ H_obs Δθ) ≤ 1e-3 standard errors (H_obs at the returned estimate).
  - A3-1(c): the gradient of the joint likelihood (a stage-1 SoD record and a stage-2 record under a random (k, r)-varying,
    95 % asymmetric window table) at θ_0 + 1e-4 s ⊙ d_j, j = 1..3. Pass iff ‖g_an − g_num‖∞ / ‖g_an‖∞ ≤ 1e-6, with g_num the
    Richardson extrapolation of central differences in ϑ at steps 1e-8 and 2e-8.
  - A3-1 is scored on the estimator's returned value under the frozen rule. Under that rule (a) passes vacuously: the returned
    value is the fallback θ_LS, flagged "not converged". (b) fails: the fallback is 8.6 standard errors from the archived ML,
    while the optimiser's own terminal iterates are within 1e-7 standard errors of it. Both facts are reported, together with
    the supplementary rule's reading of (b).
  - A3-2: codes, start, width, λ bit for bit equal to round7/results/table_T4.npz; plus z0, h, G, the designer's E[N] at σ/24,
    J_SoD_hat (bitwise evaluator_fast) and γ̄ (round-6 scan record).
  - A3-3: the static import graph of design_r8 reaches only {design_r8, declared, r8k_dr, r8k_ef, r8k_ew, r8k_td}. These use
    none of the forbidden names (θ_0 holders, every frozen module that runs the mc83 setup block, exec / eval / compile /
    __import__ / open / importlib), and import only numpy, scipy, numba, os, sys and time. A fresh `python3 -I` that imports
    design_r8 loads no forbidden module and no repository module outside that set. The sentinel positive control must raise
    on seven deliberate accesses. A guarded design must finish with an empty access log. The pipeline guards every design and
    stops if the access log is non-empty.
  - A3-4: the maximum of |Σm − 1| (≤ 1e-10) and the minimum of m (≥ 0) over every evaluator call in the checks; any violation
    raises AssertionStop.

## B — pipeline (Part B)

- DECISION B-1: N_1 and N_2 count every transmission, including the forced first one; E[N_2 | T, θ_0] and b_1 include it
  (the evaluator's E[N] = 1 + Σ). This is the convention under which B = 316.89 is E[N_SoD](θ_0) at σ/24.
- DECISION B-2: after a design failure the sequential scheme's stage 2 is the constant-Δ* SoD on the same e^(2). Its stage-2
  record is then identical to the control's. Its E[N_2 | T, θ_0] and J_table are the SoD's own σ/120 values, b_1 and J_1,
  because the policy actually run is constant Δ*. H_r uses that SoD record.
- DECISION B-3: ε_1 = max_k |h_kᵀ(θ̂_1 − θ_0)|/σ, with H from declared.H(), and per-parameter errors θ̂_1 − θ_0.
- DECISION B-4: when the two analyses have bitwise identical θ̂_1 (both fell back), they share one design, table and post-hoc
  evaluation.
- DECISION B-5: 4 worker processes with one numba thread each. Each worker runs whole replications and writes its packed
  tables. The main process appends one JSON line per finished replication to `results/archive_round8.jsonl` (fsync), and a
  re-run skips replications already archived. Every replication is kept.
- DECISION B-6: b_1 and J_1 are computed once by the reuse path at n = 120 (round 7's eval_reuse at x0). They equal round 7's
  archive bit for bit (R-POST).

## C — statistics (Part C)

- DECISION C-1: the bootstrap uses rng = default_rng(20281028); each of the 2000 resamples draws idx = rng.integers(0, R, R)
  (paired: the same idx for both schemes and all indicators). Percentile intervals are numpy.quantile at 0.025 and 0.975
  (default linear interpolation). In the subset bootstrap a resample is "不可计算" if the filtered subset has n < 10 or a
  matrix needed for that indicator has slogdet sign ≤ 0 or a non-finite value.
- DECISION C-2: the failure rates in the verdict are the design failure rate, the stage-1 estimation failure rate, and the
  joint estimation failure rate of each scheme. "设计与估计失败率各 ≤ 5 %" requires all four to be ≤ 5 %; "> 10 %" in case 3
  is triggered by any of them.
- DECISION C-3: |bias|/RMSE per parameter uses all R replications: bias_i = mean(θ̂_i − θ_0,i), RMSE_i = sqrt(M̂_ii).
- DECISION C-4: Hessian–FIM: z_ij = D̄_ij / (sd_ij / √R), with sd_ij the sample standard deviation (n − 1). The whitened
  residual is the spectral norm of J^{-1/2} D̄ J^{-1/2}, with J^{-1/2} from the symmetric eigendecomposition of
  J = (1/R) Σ J_table. As an extra diagnostic the same quantities are reported for the control's stage-2 record against J_1.
- DECISION C-5: the subset (frozen definition) for the sequential–control pair: stage-1 estimation converged, design
  succeeded, and both joint estimations converged, under the analysis's rule. Under the frozen rule it is empty unless an
  estimation is accepted.

## S — scoring readings for Q1–Q7 (written before Part B runs; the most conservative reading available)

- DECISION S-0: each prediction is scored on the primary (frozen-rule) analysis. The supplementary analysis's value is shown
  next to it, unscored. 对 iff all sub-claims hold, 部分对 if some, 错 if none.
- DECISION S-1 (Q1): (i) median ε_1 ∈ [0.2, 0.4]; (ii) 90 % quantile ≤ 0.6; (iii) fraction > 0.5 ≤ 15 %; (iv) stage-1
  estimation failure rate ≤ 2 %.
- DECISION S-2 (Q2): (i) |−log det M̂_1 − log det J_1| ≤ 0.5 nat; (ii) whitened Hessian–FIM residual ≤ 0.05.
- DECISION S-3 (Q3): (i) Pr{E[N_2|T,θ_0] > B} ≤ 10 %; (ii) Pr{N_2 > B} ≤ 25 %; (iii) b̂_seq ≤ 2B (point estimates).
- DECISION S-4 (Q4): (i) ΔD over all R ∈ [+2.5, +4] nat; (ii) bootstrap 95 % lower end ≥ +1.5.
- DECISION S-5 (Q5): (i) sequential and (ii) control |−log det M̂ − log det J| ≤ 0.5 nat.
- DECISION S-6 (Q6): (i) design failure rate ≤ 2 %; (ii) every estimation failure rate (stage 1, joint seq., joint control)
  ≤ 2 %; (iii) all 18 per-parameter |bias|/RMSE ≤ 0.2.
- DECISION S-7 (Q7): (i) every replication's primary-analysis work (stage 1 and its estimation, one design, the stage-2
  simulations, post-hoc evaluation, joint estimates; worker wall time at one numba thread with 4 workers running) ≤ 60 s;
  (ii) Part B + Part C wall time as run ≤ 6 h. The as-run total includes the supplementary analysis; a primary-only estimate
  is reported next to it.

## R — recorded after the run (no rule above was changed after Part B started)

- DECISION R-1: Part B was committed in two commits: the pipeline, statistics and verdict code while the run was going
  (88e223a, no results; a repository hook requires the working tree to be committed), then the run's outputs (3f32991).
  Round 2b set the precedent. The probe smoke tests (seeds 93000000 + r / 94000000 + r, r = 10001..10004) were run before
  the run; their outputs stay under `results/runs/` and are not committed.
- DECISION R-2: under the supplementary rule 4 of the 600 joint estimations were "not converged". In replication 134
  (sequential) and replications 98, 176 and 228 (control), both optimiser calls ended in ABNORMAL_TERMINATION_IN_LNSRCH
  (result.success false) although their Newton decrements were 1e-7 to 1e-6. The frozen "optimiser reports success"
  condition, kept in the supplementary rule, sends them to the fallback θ̂_final := θ̂_1. That is the rule as written;
  nothing was re-run.
