# Code for §8.3.0–8.3.7

All scripts assume the same setup block (system, frequencies, clock, uniform input, regressors H,
trigger `sod`, information coefficient `cfun`) which lives at the top of `mc83.py` and is loaded by
`exec(open('mc83.py').read().split('# --- noise-free')[0])`.

| file | section | what it does |
|---|---|---|
| mc83.py | setup; 8.3.6 (early R=2000) | setup block; `sod(z,D)` trigger on the clock returning event flags, pre-update reference r[k], overshoots; `cfun(al,be)` = (5.10) via `scipy.special.ndtr`; grid search (superseded) and R=2000 MC |
| sod.c | 8.3.2 | C routine: descending list of candidate thresholds -> first with N^nf >= N0. Build: `gcc -O2 -o sod sod.c -lm` |
| search_delta_star.py | 8.3.2, 8.3.4 | exact search of Delta_* over critical values |x_k-x_j| in [lo, V_T/(N0-1)]; usage `python3 search_delta_star.py 320 0.0023` |
| mc836.py | 8.3.6 | R=10000 MC, seed 20261006, common random numbers scaled by sigma_d for d in {5,20,50}: budgets, se, overshoot, S, weights; constant-output control |
| mc837.py | 8.3.7 | S, J_sod(u_unif), J_per(316), generalized eigenvalues, batch standard errors (10x1000), weakest direction, per-parameter CRLB ratios |
| mc84.py | 8.4 | MLE of the censored likelihood (8.31) by damped Newton (analytic gradient/Hessian), R=4000, seed 20261009; sd ratios vs CRLB, bias t-values, whitened eigenvalues; naive LS-on-sent-samples control. Usage `python3 mc84.py 4000` (needs wk_d20_10000.npy from mc837.py) |
| mc85_scan.py | 8.5 stage 1 | amplitude simplex scan (chi=0) and 6^4 phase scan (uniform amplitudes), CRN R=200 seed 20261010; usage `python3 mc85_scan.py k` with k=0 (amplitudes) and k=1..4 (phase quarters); outputs mc85_scan_partk.json. |
| mc85_post.py | 8.5 stage 1 | near-uniform amplitude grid, budget classification (uniform reference qualifies by definition), summary statistics, eta^2, paired batch se, lambda_min and rotation-only checks, figures; reproduces every number in section_8_5_CN.md (included) |
| mc85_predictions_before_run.md | 8.5 | predictions written before the scan |
| build.sh, tmpl_head.tex | — | pandoc + xelatex pipeline for the Chinese PDFs (Noto Serif CJK SC; lmodern.sty stripped) |
| build_en.sh | — | same for the English PDF (Liberation Serif) |

Key numbers: Delta_* = 0.002384694482635785 (N^nf = 321), sigma = Delta_*/20, B_hat = 316.87 +- 0.022 (se), N = 316.

## Stage 2 inputs (added 2026-10-10)
- mc85_stage2_protocol.md: agreed protocol and pre-run predictions for the joint 7-dim search.
- mc85_lhs.py + mc85_lhs_part{0,2..11}.json: 3000 joint candidates, R=200, seed 20261010 draws (part0 holds indices 0-499, parts 2-11 hold 500-2999 in blocks of 250).
- mc85_scan_part{0..4}.json, mc85_amp20.json: stage-1 candidates on the same draws.
- wk_d20_10000.npy: 10000-run weights of the uniform input (from mc837.py), used by mc84.py.

## Stage 2 analysis (added 2026-10-10; run from this directory, in this order; outputs in stage2/)
| file | what it does |
|---|---|
| stage2_common.py | shared setup exec'd by the stage-2 scripts: reuses mc83/mc85_scan setup, `qual` from mc85_post; numba trigger `sod_batch` proved bitwise equal to `sod()` on 100 trajectories at every start; `evaluate_crn` (reproduces stored stage-1/LHS values exactly) |
| stage2_post.py | merge 3000 LHS + 1465 stage-1 candidates (R=200, seed 20261010), dedupe, 8.3.6(g) decisions, R=200 Pareto front, predictions 1-4, selection of 40 for refinement |
| stage2_refine.py | the 40 at R=2000, seed 20261012 (same draws incl. reference): D +- se, decisions, Delta m_D with paired batch se, eigenvalues, re-ranking |
| stage2_local.py | penalised Nelder-Mead (R=500 inside, maxfev=150, 4 targets x 3 front regions); every new point re-evaluated at R=2000, seed 20261012; phase-only controls |
| stage2_final.py | 7 final candidates: batch 1 (seed 20261013, R=4000) J, D, eigenvalues, worst combination, q_k/x_k/w_k, gaps, powers; batch 2 (seed 20261014, R=4000) ML covariance check with mc84.py functions |
| stage2_figs.py | figures for stage2/stage2_report.md |
| stage2_followup.py | report §8: budget margins and an independent decision re-run (seed 20261015), independent ML bias check for NM-none-P5514 and AB-0.5 (seed 20261016, batch-2 code of stage2_final.py exec'd verbatim), window vs long-run input power |
| stage2/stage2_report.md | report; every number traceable to stage2/*.log |

## Stage 4 (added 2026-10-13; run from this directory, in this order; outputs in stage4/)
| file | what it does |
|---|---|
| stage4/predictions.md, stage4/design.md | predictions and every rule not fixed by the prompt, committed before any stage-4 run |
| stage4_common.py | bookkeeping only (stage-2/3 pool loader, used-start keys, feasibility rule, representative rule); exec's stage3_common.py unchanged |
| stage4_search.py | Script 1: 12 new NM starts per floor (6 front by m_E stratum, 6 from the qualified R=200 pool, seed 20261018), maxfev 300, R=500 inside; 10 unbudgeted starts; endpoint + constrained archive best per run; everything at R=2000 (seed 20261012); merged front, per-floor and unbudgeted previous vs new best |
| stage4_confirm.py | Script 2: `freeze` (pool, incumbents, delta, method, family, M, t; committed before the draw), then `draw` (seed 20261020, R=8000, 20x400 batch-mean t-intervals with t_{19,1-0.05/(2M)}); classification and candidate near-optimal sets |
| stage4_scenarios.py | Script 3: the four floor representatives + UNIF at d=5 and d=50 (and a d=20 control), seed 20261021, R=4000 CRN; sigma_d, Jb, ldB rebound per scenario |
| stage4_figs.py | report figures and derived checks from the JSON outputs |
| stage4/stage4_report.md | report; every number traceable to stage4/*.log |

## Route 1: free-weight D-optimal relaxation (added 2026-10-14; run from anywhere, in this order; outputs in bound/)
| file | what it does |
|---|---|
| bound/predictions.md, bound/review_note.md | predictions committed before any run; review changes (outer search dropped, B at the input's own S(u), cutting-plane certificates, wording) |
| bound/bound_common.py | shared setup (mc83/mc85_scan setup without its draws; stage-2 parametrisation); certificates (cutting-plane upper value U, duality bound); own solvers: batched pairwise exchange (numba) and spectral projected gradient; cvxpy log_det wrapper |
| bound/bound_inner.py | Script 1: max log det(sigma^-2 sum xi_k h_k h_k^T) over {0 <= xi <= cap, sum xi = W} at the uniform input: A (W=316, cap 1), B (W=S(u_unif)=465.958, cap 1), C (W=316, no cap); cross-checks SPG, SCS, Clarabel; equivalent sample counts; fig_inner_A.png |
| bound/bound_dozzi.py | Script 2: Dozzi et al. (arXiv:2605.13120) D-optimal density (eq. 15) on the clock grid, with a cutting-plane bound for the continuous problem, and their greedy sequence (Sec. 4.1) at N=316; fig_dozzi.png |
| bound/bound_identity.py | Script 3: H(u) = H(u_unif) T(u) identity (random fixed xi, seed 20261022; direct solves at 11 inputs) and the free-placement relaxation gap of the 7 stage-2 finalists at their own S(u) |
| bound/bound_report.md | report; every number traceable to bound/*.log |
Needs cvxpy (with SCS, Clarabel) and numba in addition to requirements.txt.

## Evaluator acceptance (added 2026-10-19; run from evaluator/, in this order; outputs in evaluator/logs and evaluator/results)
Deterministic evaluator u -> (E[N], S, J) by the 1-D density recursion of the reference value (spec `reports_CN/plan_evaluator_CN.md` v4.1, frozen).
| file | what it does |
|---|---|
| evaluator/make_config.py -> config.json | step 0: the four test inputs U, D, E, N (full a_m, chi_m read from the stored JSON, with source log lines), grids n = 6/12/24, margins 8/16 Delta, constants, MC seed/R/B, hardware |
| evaluator/predictions.md | step 1: spec §8 P1-P12 and the §9 acceptance table, verbatim, committed before any run |
| evaluator/density_recursion_v1.py | step 2: `--version v0` (prototype density_recursion.py, numpy, bitwise), `A`, `B` (mass formulation, numba, prefix-sum windows); `--test` runs unit tests (a), (b), (c) -> logs/unit_tests.log |
| evaluator/run_all.py | step 3: 3 versions x 4 inputs x 3 grids + tail check (B, n=12, 16 Delta), serial and timed; c invariants on every node; §6 trigger check (`--n48 X` only after a trigger) -> logs/run_all.log, results/ |
| evaluator/mc_reference.py | step 4: MC seed 20261019, R = 10000 = 20 x 500, CRN over the four inputs; bitwise check against stage2_common.evaluate_crn; old-seed U check -> logs/mc_reference.log, results/mc/ |
| evaluator/analysis.py | step 5: convergence, B stability, tail, 199-item whitened MC test, runtime, §9 criteria, prediction evidence -> logs/analysis.log, results/analysis.json |
| evaluator/report_tables.py, profile_runtime.py | report tables generated from the stored results (logs/report_tables.md); runtime split of one B evaluation (logs/profile_runtime.log) |
| evaluator/report_evaluator_CN.md, REVIEW_GUIDE.md, build_review_zip.py | step 6: report (every number traceable to evaluator/logs/*.log), review guide, review package `review_evaluator_8_5.zip` at the repo root |

Trigger conventions, matched to `mc83.sod` (and `stage2_common.sod_batch`, bitwise equal to it):
- a send happens iff |z_k - r_k| >= Delta (">=", not ">"); z_k = x_k + sigma e_k is the noisy output;
- k = 1 is a forced send: the recursion starts from f_2 = g_1 (A/B: m_{2,j} = ghat_{1,j}), E[N] starts at 1 and w_1 = 1;
- the reference after a send is the noisy z_k itself (in the recursion the new reference is the measurement node j);
- P_k(r) = Phi(beta) - Phi(alpha) is the no-send probability of the open band |z - r| < Delta. Version A counts the node at
  |i - j| = M (= Delta/h) as a send; version B gives it weight 1/2, the trapezoid rule for the band integral (the physical
  trigger is still pointwise >= Delta).

Reproduce every table of `evaluator/report_evaluator_CN.md` (needs numpy, scipy, numba; about 7 minutes):

    cd evaluator
    python3 density_recursion_v1.py --test
    python3 run_all.py            # resumable: finished evaluations in results/ are skipped; delete results/rec_* to recompute
    python3 mc_reference.py
    python3 analysis.py
    python3 report_tables.py > logs/report_tables.md
    python3 profile_runtime.py N 24 > logs/profile_runtime.log

Every number of the report is on a log line: `EV` (run_all.log), `MC` / `OLD-U` (mc_reference.log), `CONV`, `STAB`, `TAIL`,
`ITEM`, `SIGN`, `NONLIN`, `DISPLAY`, `APPX`, `RUNTIME`, `SEARCH-ROUND`, `CRIT`, `P1`-`P9` (analysis.log), `PROFILE`
(profile_runtime.log). The evaluator's numbers are deterministic; runtimes depend on machine load.

## Evaluator window speed-up, version B only (added 2026-10-19; run from evaluator/; spec `reports_CN/plan_speedup_CN.md` v3.1)
| file | what it does |
|---|---|
| evaluator/predictions_speedup.md | S1-S5 of spec §6, verbatim, committed before the kernel test and any run |
| evaluator/kernel_zero_test.py | Step 0: the accepted `_kernel` (same compiled function) is exactly 0.0 at 39, 40, 50 sigma from every x_k of U, D, E, N -> rho = 39 sigma (logs/kernel_zero_test.log, config_fast.json) |
| evaluator/evaluator_fast.py | the window path: `run_window` (B only), `evaluate_fast` (the accepted `evaluate` with its `run_ab` swapped for one call), `--windows` geometry check, `--selftest` synthetic check of the assertion; imports density_recursion_v1.py unchanged, whose full-grid path stays callable |
| evaluator/regression_fast.py | Step 4: the 12 accepted B evaluations, unit tests (b), (c) for B, tail check, assertion count, node counts; exit 1 on any failure (logs/speedup_regression.log, results/fast/) |
| evaluator/timing_fast.py, profile_fast.py | Step 5: 12 evaluations x 3, medians, 20000-evaluation search cost (logs/speedup_timing.log); runtime split (logs/speedup_profile.log) |
| evaluator/report_speedup_CN.md, config_fast.json | report; rho, R, displacement bound, frozen tolerances, hardware |

Window: W_k = {|z - x_k| < R}, R = Delta + 48 sigma = Delta + rho (39 sigma) + displacement bound (5 sigma) + margin (4 sigma);
step k works on the node range [a_k, b_k] = [ceil((x_k - R - z_0)/h), floor((x_k + R - z_0)/h)], prefix and window sums on
[a_k - M, b_k + M], both clipped to the grid. Grid, kernel, scheme, cfun and update order are the accepted ones.

Induction invariant (spec §2). The kernel is exactly 0.0 beyond rho (Step 0), so one update gives supp(m_{k+1}) inside
|z - x_k| < Delta + rho; with |x_{k+1} - x_k| <= 5 sigma this is inside |z - x_{k+1}| < Delta + 44 sigma, inside W_{k+1}.
Start: supp(m_2) = supp(ghat_1) lies within rho of x_1 and |x_2 - x_1| <= 5 sigma, so inside W_2 (asserted at initialisation).

Assertion and its dependence on the invariant. After the update of step k the destination buffer is checked for any non-zero
value (not a signed sum) on the M nodes on each side of W_{k+1}. This is enough only because of the invariant and the buffer
rule: before step k the buffer is zero outside W_{k-1} (its old m_{k-1}), it is zeroed on W_{k-1} and written on W_k only, so
a non-zero outside W_{k+1} can only sit in W_k \ W_{k+1}, which lies within 5n < M nodes of W_{k+1}. It is not an independent
check of the whole array. On failure the step is redone on the full grid from the untouched source buffer (the pre-update
m_k), the step's E[N] increment, w_k and mass entry are replaced, the event is logged, and the whole run stops (diagnostic only).

Last step. As in the accepted version, the update is also done at k = K (m_{K+1} is not used); the assertion then checks that
the support after the update lies inside W_K (there is no x_{K+1}, no W_{K+1}); the displacement column reads
"not applicable — no next sample". No sample point is added.

Run the regression (from evaluator/; numpy, scipy, numba):

    python3 kernel_zero_test.py && python3 evaluator_fast.py --windows && python3 evaluator_fast.py --selftest
    python3 regression_fast.py        # last line "REGRESSION PASS ..."; exit code 1 and "STOP" on any failure or assertion
    python3 timing_fast.py            # only after the regression passed

The regression compares against the accepted stored outputs results/rec_B_*_m8.json, J_B_*.npy, w_B_*.npy with the tolerances
in config_fast.json (relative 1e-12 for E[N], S, log det; |Delta| <= 1e-12 (|ref| + max|J_ref|) for J; |Delta| <= 1e-12 (|w_ref| + 1)
for w_k; own mass drift <= 1e-10; tail check with 1e-11) and requires 0 assertion failures.
