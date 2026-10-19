"""Window speed-up of the accepted evaluator, version B only (spec reports_CN/plan_speedup_CN.md v3.1, frozen).

The accepted evaluator density_recursion_v1.py is imported, not modified: its grid (z_0, h, G), inputs, kernel expression,
discrete scheme, cfun use, record and timing code are shared, and its full-grid path stays callable unchanged (for the
fallback of spec §4 and for future regressions). The only change is the index range each step works on.

Window (spec §2-§3). rho = 39 sigma (kernel exactly 0.0 beyond it: logs/kernel_zero_test.log), displacement bound 5 sigma
(family bound v_max = dt sqrt(sum_m (w_m |G(j w_m)|)^2) = 4.57 sigma, rounded up), margin 4 sigma: R = Delta + 48 sigma.
Step k = 2..K works on the closed index range [a_k, b_k], a_k = ceil((x_k - R - z_0)/h), b_k = floor((x_k + R - z_0)/h),
clipped to [0, G-1].

Usage (from this directory):  python3 evaluator_fast.py --windows        # window geometry check -> logs/speedup_windows.log
                              python3 evaluator_fast.py --selftest       # synthetic checks of the assertion and fallback step
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


# --- the windowed step (spec §3), numba with the accepted flags (@njit(cache=True): no fastmath, no parallel) ----------
from numba import njit

@njit(cache=True)
def _ab_step_w(m, mn, gh, Cg, Cm, M, wb, a, b, ap, bp, an, bn):
    """one step of spec §3 on [a, b] = [a_k, b_k]; m = source buffer (m_k, not modified), mn = destination buffer.
    Prefix sums of ghat and m on [lo, hi] = [a-M, b+M] (clipped) with the accepted _prefix, stored at their global indices
    (C[lo] = 0: everything left of lo is exactly zero) and C[G] := C[hi+1], so that the accepted _window reads them unchanged.
    Buffer rule: the destination is zeroed on [ap, bp] = [a_{k-1}, b_{k-1}] (the only place it holds non-zeros, m_{k-1}),
    then written on [a, b] only. Assertion (spec §4): mn == 0 on the M nodes on each side of [an, bn] (W_{k+1}; W_K at k = K).
    Returns (sum_i m_i (1-Phat_i), sum mn, min mn, first and last j with mn_j != 0, number of non-zero mn_j, assertion ok),
    the sums, min and support taken over [a, b]."""
    ng = m.size; lo = max(a-M, 0); hi = min(b+M, ng-1)
    dr._prefix(gh[lo:hi+1], Cg[lo:hi+2]); dr._prefix(m[lo:hi+1], Cm[lo:hi+2])
    Cg[ng] = Cg[hi+1]; Cm[ng] = Cm[hi+1]
    for j in range(ap, bp+1): mn[j] = 0.0
    en = 0.0
    for i in range(a, b+1):
        pin, pout = dr._window(gh, Cg, i, M, wb)
        _, T = dr._window(m, Cm, i, M, wb)
        en += m[i]*pout
        mn[i] = m[i]*pin+gh[i]*T
    tot = 0.0; mmin = mn[a]; slo = -1; shi = -1; nnz = 0
    for j in range(a, b+1):
        v = mn[j]; tot += v
        if v < mmin: mmin = v
        if v != 0.0:
            nnz += 1
            if slo < 0: slo = j
            shi = j
    ok = True
    for j in range(max(an-M, 0), an):
        if mn[j] != 0.0: ok = False
    for j in range(bn+1, min(bn+M, ng-1)+1):
        if mn[j] != 0.0: ok = False
    return en, tot, mmin, slo, shi, nnz, ok


def full_step(m_src, z, x_now, dz, M, wb):
    """the accepted full-grid step (dr._kernel, dr._ab_step on fresh full-size arrays) from the source buffer m_k; used only
    by the fallback of spec §4 (diagnostic) and by the self-test. Returns kmass, (en, mass, min m, lo, hi), m_{k+1}."""
    ng = z.size; gh = np.empty(ng); mn = np.empty(ng); Cg = np.empty(ng+1); Cm = np.empty(ng+1)
    km = dr._kernel(z, x_now, s, dz, gh)
    return km, dr._ab_step(m_src, mn, gh, Cg, Cm, M, wb), mn


class AssertionStop(RuntimeError):
    """raised after the spec §4 fallback (or a failed initialisation check): the run stops, nothing later may use it."""


def run_window(xk, n, margin):
    """version B on the window path; same outputs as density_recursion_v1.run_ab plus the per-step window records."""
    z, dz = dr.make_grid(xk, n, margin, 'B'); ng = len(z); M = 20*n; wb = dr.WB['B']
    assert abs(D/dz-M) < 1e-6, (D/dz, M)
    a, b, _, _ = windows(xk, z[0], dz, ng)                      # array index p <-> step p+1: a[p], b[p] bound W around x[p]
    m = np.zeros(ng); mn = np.zeros(ng); gh = np.zeros(ng); Cg = np.empty(ng+1); Cm = np.empty(ng+1)
    EN = 1.0; ENP = 1.0; W = np.zeros(K); W[0] = 1.0
    mass = np.empty(K); minm = np.empty(K); kmass = np.empty(K)
    nnz = np.zeros(K, np.int64); slo = np.zeros(K, np.int64); shi = np.zeros(K, np.int64); ok = np.ones(K, np.bool_)
    # initialisation m_2 = ghat_1: the accepted full-grid kernel once, support asserted inside W_2, written into W_2 only
    g1 = np.empty(ng); kmass[0] = dr._kernel(z, xk[0], s, dz, g1)
    nz = np.flatnonzero(g1); ok[0] = bool(nz[0] >= a[1] and nz[-1] <= b[1])
    if not ok[0]:
        raise AssertionStop(f"initialisation: support of m_2 [{nz[0]}, {nz[-1]}] not inside W_2 [{a[1]}, {b[1]}]; run stopped")
    m[a[1]:b[1]+1] = g1[a[1]:b[1]+1]
    nz = np.flatnonzero(m); lo, hi = nz[0], nz[-1]
    mass[0] = m.sum(); minm[0] = m.min(); nnz[0] = nz.size; slo[0], shi[0] = lo, hi
    for k in range(1, K):
        zz = z[lo:hi+1]; mm = m[lo:hi+1]
        al = (zz-D-xk[k])/s; be = (zz+D-xk[k])/s
        cev, cno = dr.cfun(al, be)
        W[k] = mm@(cev+cno)
        P = np.where(al > 0, dr.ndtr(-al)-dr.ndtr(-be), dr.ndtr(be)-dr.ndtr(al))     # as in run_ab
        ENP += mm@(1-P)
        gh[a[k-1]:b[k-1]+1] = 0.0                                                   # kernel buffer: previous window cleared
        kmass[k] = dr._kernel(z[a[k]:b[k]+1], xk[k], s, dz, gh[a[k]:b[k]+1])        # accepted kernel on [a_k, b_k]
        an, bn = (a[k+1], b[k+1]) if k < K-1 else (a[k], b[k])                      # W_{k+1}; at the last step W_K (spec §3)
        en, mass[k], minm[k], lo, hi, nnz[k], ok[k] = _ab_step_w(m, mn, gh, Cg, Cm, M, wb, a[k], b[k], a[k-1], b[k-1], an, bn)
        if not ok[k]:
            # spec §4: redo this step on the full grid from the untouched source buffer (the pre-update m_k), replace the
            # step's E[N] increments, w_k and mass entry (no double counting), log, and stop the whole run
            nzs = np.flatnonzero(m); fl, fh = nzs[0], nzs[-1]
            zf = z[fl:fh+1]; mf = m[fl:fh+1]; alf = (zf-D-xk[k])/s; bef = (zf+D-xk[k])/s
            cevf, cnof = dr.cfun(alf, bef); W[k] = mf@(cevf+cnof)
            Pf = np.where(alf > 0, dr.ndtr(-alf)-dr.ndtr(-bef), dr.ndtr(bef)-dr.ndtr(alf))
            ENP = ENP-mm@(1-P)+mf@(1-Pf)
            km, (enf, massf, minf, _, _), _ = full_step(m, z, xk[k], dz, M, wb)
            kmass[k], mass[k], minm[k] = km, massf, minf
            print(f"ASSERT-FAIL step {k+1} (array index {k}) n={n}: non-zero mass within M nodes outside "
                  f"[{an}, {bn}]; full-grid redo from the pre-update buffer: E[N] increment fast {en!r} full {enf!r}, "
                  f"w_k {W[k]!r}, mass {massf!r}; RUN STOPPED")
            raise AssertionStop(f"assertion failed at step {k+1}; run stopped (spec §4)")
        EN += en; m, mn = mn, m
    disp = np.full(K, np.nan); disp[:-1] = np.abs(np.diff(xk))/s                   # |x_{k+1} - x_k|/sigma; none at k = K
    return dict(EN=EN, ENP=ENP, W=W, mass=mass, minm=minm, kmass=kmass, G=ng, M=M, Delta_over_h=D/dz,
                win_a=a, win_b=b, nnz=nnz, sup_lo=slo, sup_hi=shi, assert_ok=ok, disp=disp)


def evaluate_fast(name, n, margin_mult=8):
    """the accepted dr.evaluate('B', ...) -- same record, J assembly, invariants and steady-state timing -- with its full-grid
    run_ab replaced by run_window for this call only (dr.run_ab is restored afterwards, so the accepted path is unchanged)."""
    full = dr.run_ab
    dr.run_ab = lambda xk, n_, margin, version: run_window(xk, n_, margin)
    try: return dr.evaluate('B', name, n, margin_mult)
    finally: dr.run_ab = full

def _warmup():
    dr._warmup()
    z = np.linspace(-1.0, 1.0, 201); m = np.zeros(201); m[90:110] = 0.05; mn = np.zeros(201); gh = np.zeros(201)
    dr._kernel(z[50:150], 0.0, 0.1, z[1]-z[0], gh[50:150])
    _ab_step_w(m, mn, gh, np.empty(202), np.empty(202), 10, 0.5, 50, 149, 50, 149, 50, 149)


def selftest():
    """synthetic checks of the safety mechanism, separate from any run (spec §4): (1) the assertion inside _ab_step_w fires
    when non-zero mass lies within M nodes outside the next window and stays silent otherwise; (2) on the first step of
    U, n = 6, the windowed step equals the accepted full-grid step (full_step) on m_{k+1}, E[N] increment, mass, min m, support."""
    sys.stdout = dr.Tee(os.path.join(LOGS, 'speedup_selftest.log'))
    _warmup()
    print(f"=== evaluator_fast.py --selftest (synthetic checks of the step-2 safety mechanism; not a run)")
    xk, _ = dr.input_arrays('U'); n = 6; z, dz = dr.make_grid(xk, n, 8*D, 'B'); ng = len(z); M = 20*n
    a, b, _, _ = windows(xk, z[0], dz, ng)
    g1 = np.empty(ng); dr._kernel(z, xk[0], s, dz, g1); m = np.zeros(ng); m[a[1]:b[1]+1] = g1[a[1]:b[1]+1]
    gh = np.zeros(ng); dr._kernel(z[a[1]:b[1]+1], xk[1], s, dz, gh[a[1]:b[1]+1])
    res = {}
    for lab, (an, bn) in (('true next window W_3', (a[2], b[2])), ('next window moved 2M nodes right', (a[2]+2*M, b[2]+2*M))):
        mn = np.zeros(ng)
        res[lab] = _ab_step_w(m, mn, gh, np.empty(ng+1), np.empty(ng+1), M, 0.5, a[1], b[1], a[0], b[0], an, bn)
        print(f"SELFTEST (1) assertion with {lab}: ok = {res[lab][6]}")
    mn = np.zeros(ng); r = _ab_step_w(m, mn, gh, np.empty(ng+1), np.empty(ng+1), M, 0.5, a[1], b[1], a[0], b[0], a[2], b[2])
    km, rf, mnf = full_step(m, z, xk[1], dz, M, 0.5)
    same = (np.array_equal(mn, mnf) and r[0] == rf[0] and r[1] == rf[1] and r[2] == rf[2] and (r[3], r[4]) == (rf[3], rf[4]))
    print(f"SELFTEST (2) first step of U, n = 6: windowed == accepted full-grid (m_3 bitwise, E[N] increment, mass, min, support): {same}; "
          f"E[N] increment {r[0]!r} vs {rf[0]!r}; mass {r[1]!r} vs {rf[1]!r}")
    ok = res['true next window W_3'][6] and not res['next window moved 2M nodes right'][6] and same
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


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
    ap = argparse.ArgumentParser(); ap.add_argument('--windows', action='store_true'); ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.windows: window_check()
    if args.selftest: sys.exit(0 if selftest() else 1)
