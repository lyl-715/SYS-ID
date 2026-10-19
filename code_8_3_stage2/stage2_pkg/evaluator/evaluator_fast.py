"""Window speed-up of the accepted evaluator, version B only (spec reports_CN/plan_speedup_CN.md v3.1, frozen).

The accepted evaluator density_recursion_v1.py is imported, not modified: its grid (z_0, h, G), inputs, kernel expression,
discrete scheme, cfun use, record and timing code are shared, and its full-grid path stays callable unchanged (for the
fallback of spec §4 and for future regressions). The only change is the index range each step works on.

Window (spec §2-§3). rho = 39 sigma (kernel exactly 0.0 beyond it: logs/kernel_zero_test.log), displacement bound 5 sigma
(family bound v_max = dt sqrt(sum_m (w_m |G(j w_m)|)^2) = 4.57 sigma, rounded up), margin 4 sigma: R = Delta + 48 sigma.
Step k = 2..K works on the closed index range [a_k, b_k], a_k = ceil((x_k - R - z_0)/h), b_k = floor((x_k + R - z_0)/h),
clipped to [0, G-1].

Usage (from this directory):  python3 evaluator_fast.py --windows        # window geometry check -> logs/speedup_windows.log
"""
import os, sys, json, math, argparse
import numpy as np
import density_recursion_v1 as dr

HERE = dr.HERE; LOGS = dr.LOGS
CFGF = json.load(open(os.path.join(HERE, 'config_fast.json')))
s, D, K = dr.s, dr.D, dr.K
RHO = 39*s; DISP = 5*s; MARGIN = 4*s
R = D+48*s                                           # spec §2: R := Delta + 48 sigma = Delta + rho + 5 sigma + 4 sigma
INPUTS = ['U', 'D', 'E', 'N']; GRIDS = dr.CFG['grids']['n']


def v_max():
    """family bound on |x_{k+1} - x_k| (spec §2 lemma 3): dt sqrt(sum_m (w_m |G(j w_m)|)^2), Cauchy-Schwarz over sum a_m^2 = 1."""
    return dr.dt*float(np.sqrt(np.sum((dr.w*np.abs(dr.Gj))**2)))

def windows(xk, z0, h, G):
    """closed index ranges [a_k, b_k] of the window W_k = {z : |z - x_k| < R} for every k (array index k-1 <-> step k),
    clipped to [0, G-1]; also the unclipped ends (to log whether clipping ever happened)."""
    a = np.ceil((xk-R-z0)/h).astype(np.int64); b = np.floor((xk+R-z0)/h).astype(np.int64)
    return np.clip(a, 0, G-1), np.clip(b, 0, G-1), a, b


def window_check():
    """spec §2-§3 geometry for the four inputs and three grids (margin 8 Delta) and the tail-check grid (n = 12, 16 Delta)."""
    sys.stdout = dr.Tee(os.path.join(LOGS, 'speedup_windows.log'))
    vm = v_max()
    print(f"=== evaluator_fast.py --windows; spec {CFGF['spec']}; R = Delta + 48 sigma = {R!r} = {R/s!r} sigma; rho = {RHO/s} sigma; "
          f"displacement bound {DISP/s} sigma; margin {MARGIN/s} sigma")
    print(f"WIN family bound v_max = dt sqrt(sum (w_m |G(j w_m)|)^2) = {vm!r} = {vm/s!r} sigma (spec: 5.45e-4 = 4.57 sigma; rounded bound 5 sigma)")
    for name in INPUTS:
        xk, _ = dr.input_arrays(name); dx = np.abs(np.diff(xk))/s
        print(f"WIN {name} actual max_k |x_(k+1) - x_k| = {float(dx.max())!r} sigma (check item, not used in the window); "
              f"<= family bound {vm/s:.4f} sigma: {bool(dx.max() <= vm/s)}")
        for n, mm in [(6, 8), (12, 8), (24, 8), (12, 16)]:
            z, h = dr.make_grid(xk, n, mm*D, 'B'); G = len(z)
            a, b, a0, b0 = windows(xk, z[0], h, G); cnt = b-a+1
            print(f"WIN {name} n={n} {mm}Delta: G = {G}, M = {20*n}; window nodes b_k - a_k + 1 min {int(cnt.min())} max {int(cnt.max())} "
                  f"(spec bound 2*68*n + 1 = {2*68*n+1}); clipped at the grid ends: {int((a0 < 0).sum()+(b0 > G-1).sum())} steps; "
                  f"min a_k - M {int((a-20*n).min())}, max b_k + M {int((b+20*n).max())} (G - 1 = {G-1})")
    cfg = dict(CFGF)
    cfg.update(R=dict(value=R, definition='Delta + 48 sigma = Delta + rho (39 sigma) + displacement bound (5 sigma) + margin (4 sigma)'),
               displacement_bound=dict(value=DISP, definition='5 sigma: family bound v_max rounded up', v_max_computed=vm, v_max_over_sigma=vm/s),
               margin=dict(value=MARGIN, definition='4 sigma'),
               window='[a_k, b_k], a_k = ceil((x_k - R - z_0)/h), b_k = floor((x_k + R - z_0)/h), clipped to [0, G-1]; '
                      'prefix sums and window sums on [a_k - M, b_k + M] (clipped); kernel normalisation on [a_k, b_k]',
               tolerances=dict(
                   regression=dict(positive_quantities='E[N] (P and Phat), S, log det J: |fast - ref|/|ref| <= 1e-12',
                                   J_elements='|Delta| <= 1e-12 (|ref| + max|J_ref|), all 81 elements',
                                   w_k='|Delta| <= 1e-12 (|w_k,ref| + 1), all 5000 values',
                                   mass='fast version own per-step max |sum m - 1| <= 1e-10 (residual not compared between versions)'),
                   tail_check='B, n = 12, 16 Delta vs 8 Delta, both on the fast path: the same quantities and classes with 1e-12 replaced by 1e-11',
                   unit_tests='(b) and (c) on the fast path: unchanged within the regression tolerances; (b) |B - 0.5| <= 1e-6, (c) mass <= 1e-10, min m >= 0',
                   assertion_count='0 over 12 x 4999 steps (acceptance criterion)'))
    json.dump(cfg, open(os.path.join(HERE, 'config_fast.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--windows', action='store_true')
    args = ap.parse_args()
    if args.windows: window_check()
