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
