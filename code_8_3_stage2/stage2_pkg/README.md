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
