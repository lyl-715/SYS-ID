"""Round 8, Part A2: the stage-2 design module T = T(Y_1) (spec reports_CN/plan_round8_CN.md v3.1, frozen; §1.3, §1.7).

Inputs: theta_hat_1 and the declared knowledge (declared.py) ONLY. This module imports numpy, scipy, numba, declared and the
generated verbatim mirrors r8k_dr / r8k_ef / r8k_ew / r8k_td (build_kernels.py) -- nothing that holds theta_0 (static
import-graph check and runtime sentinel in isolation_r8.py / checks_r8.py).

design(theta_hat) runs, all at theta_hat:
  1. nominal output x_hat_k = h_k^T theta_hat (declared.x_polar); non-finite -> design failure;
  2. the design grid, support windows, kernels: round 4's Design (verbatim) on x_hat -- grid origin z_0 = min x_hat - 8 Delta,
     h = sigma/24, support start indices / widths from x_hat (no module-level constant of the original nominal model; every
     coordinate is generated here and returned with the table);
  3. J_SoD_hat: the accepted evaluator's window path (evaluate_fast, sigma/24, constant Delta*) at x_hat; A = J_SoD_hat^-1;
     gamma_k = sigma^-2 h_k^T A h_k by round 4's Design.gamma('W') (Cholesky solve, verbatim); gbar = mean_{k>=2} gamma_k;
  4. lambda = 4 gbar (declared.design_lambda); DP (round 4's Design.dp, verbatim) -> deterministic policy (codes);
  5. the designer's own expectation E_hat[N_2 | T, theta_hat]: the general window path at sigma/24 on the design grid.

Design failures that go to the frozen fallback (spec §1.7) -- returned as status 'design_failed' with the reason, nothing
else: J_SoD_hat non-finite or Cholesky failure; non-finite A, gamma_k, gbar or nominal output; a numerical exception in the
DP (ArithmeticError) or no selectable action (a node whose Bellman maximum is not finite). Every other exception -- in
particular the conservation / non-negativity / support assertions of the evaluators (AssertionStop, AssertionError) --
propagates: correctness errors stop the run.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
import declared as dk
import r8k_dr, r8k_ef, r8k_ew, r8k_td

K = dk.K


class DesignFailure(Exception):
    """one of the pre-listed numerical design failures of spec §1.7 (goes to the fallback)."""


def warmup():
    r8k_td._warmup()


def design(theta_hat, tag='design', log=None, keep_design=False):
    """the design of spec §1.3 at theta_hat. Returns a dict: status ('ok' | 'design_failed'), reason, and for 'ok' the table
    (codes uint16 (K-1, WMAX), start, width, z0, h, G -- the coordinates of this design), lambda, gbar, gamma_1, gamma (K,),
    J_sod_hat, A, E_hat[N_2 | T, theta_hat] and the designer's other sigma/24 quantities, timings."""
    log = log or (lambda m: None)
    th = np.array(theta_hat, float)
    out = dict(status='ok', reason=None, theta_hat=th.tolist())
    t0 = time.perf_counter()
    try:
        xh = dk.x_polar(th)
        if not np.all(np.isfinite(xh)): raise DesignFailure('nominal output non-finite')
        r8k_dr._INPUTS[tag] = (xh, dk.H())
        des = r8k_td.Design(tag, log)                                   # grid, support windows, kernel cache (verbatim)
        out['t_cache_s'] = des.cache_s
        # J_SoD_hat, A, gamma (spec §1.3); Cholesky failure / non-finite -> design failure
        rec_sod, J_sod, W_sod = des.sod_reference()
        if not np.all(np.isfinite(J_sod)): raise DesignFailure('J_SoD_hat non-finite')
        try:
            cf = cho_factor(J_sod, lower=True)
        except LinAlgError as e:
            raise DesignFailure(f'Cholesky failure of J_SoD_hat ({e})')
        A = cho_solve(cf, np.eye(dk.N_PAR))
        if not np.all(np.isfinite(A)): raise DesignFailure('A = J_SoD_hat^-1 non-finite')
        try:
            gam, gbar, g1 = des.gamma('W')                                # round 4 verbatim: Cholesky solve, no explicit inverse
        except LinAlgError as e:
            raise DesignFailure(f'Cholesky failure of J_SoD_hat in gamma ({e})')
        if not (np.all(np.isfinite(gam)) and np.isfinite(gbar)): raise DesignFailure('gamma_k or gbar non-finite')
        lam = dk.design_lambda(gbar)
        if not np.isfinite(lam): raise DesignFailure('lambda non-finite')
        # DP at fixed lambda (round 4 verbatim); numerical exception or a node without a finite maximum -> design failure
        codes = np.empty((K-1, des.WMAX), np.uint16)
        t1 = time.perf_counter()
        try:
            dsum, nbad = des.dp(gam, lam, codes)
        except ArithmeticError as e:
            raise DesignFailure(f'numerical exception in the DP ({type(e).__name__}: {e})')
        out['t_dp_s'] = time.perf_counter()-t1
        if nbad > 0: raise DesignFailure(f'no selectable action: {nbad} node-steps without a finite Bellman maximum')
        if not np.isfinite(dsum): raise DesignFailure('DP value non-finite')
    except DesignFailure as e:
        r8k_dr._INPUTS.pop(tag, None)
        out.update(status='design_failed', reason=str(e), t_total_s=time.perf_counter()-t0)
        log(f"DESIGN {tag} FAILED (spec §1.7 fallback): {e}")
        return out
    # the designer's own expectation (general path, sigma/24, design grid); assertions propagate
    t1 = time.perf_counter()
    tab = des.table(codes)
    rec, J_hat, rr = r8k_ew.evaluate_table(tag, tab)
    out['t_fwd_s'] = time.perf_counter()-t1
    F = float(gam[1:]@rr['W'][1:])
    out.update(codes=codes, start=des.a[1:].copy(), width=des.w[1:].copy(), z0=float(des.z[0]), h=float(des.dz), G=int(des.G),
               WMAX=int(des.WMAX), lam=float(lam), gbar=float(gbar), gamma1=float(g1), gamma=gam, J_sod_hat=J_sod, A=A,
               EN_sod_hat=float(rec_sod['EN']), EN_hat=float(rec['EN']), S_hat=float(rec['S']), logdet_hat=float(rec['logdet']),
               F_hat=F, dual_part=float(dsum), J_hat=J_hat, mass_dev_max_hat=float(rec['mass_dev_max']), min_m_hat=float(rec['min_m']),
               mass_dev_max_sod_hat=float(rec_sod['mass_dev_max']), min_m_sod_hat=float(rec_sod['min_m']),
               xhat_min=float(xh.min()), xhat_max=float(xh.max()), path_hat=rec['path'], t_total_s=time.perf_counter()-t0)
    if keep_design: out['des'] = des
    r8k_dr._INPUTS.pop(tag, None)
    return out
