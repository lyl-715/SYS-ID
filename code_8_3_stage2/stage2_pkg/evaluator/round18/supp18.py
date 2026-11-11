"""Round 18, SUPPLEMENTARY analysis of Part B (user instruction of 2026-11-11 after stage B; DECISION 18-S1 .. 18-S4 of the
report). It does NOT replace the pre-registered estimator of predictions_round18.md (DECISION 18-6: estimator_r8b.estimate_stage1,
fallback theta_LS for a run whose two L-BFGS-B calls are both not accepted); every pre-registered number stays as it is.

For the CML runs of stage B that were not accepted (partB.json -> triggers.<T>.not_accepted_runs), and for those only:
  1. the record of experiment i is rebuilt exactly as in stage B (acc18.noise_B, acc18.simulate_B, acc18.stage_data) and
     estimator_r8b.estimate_stage1(data, 80261116 + i) is re-run; it must reproduce stage B's decision (not accepted, fallback
     theta_LS bitwise equal to the archived theta_LS);
  2. start point: the end point of the FIRST L-BFGS-B call (the call started from theta_LS; DECISION 18-S1);
  3. full Newton steps v <- v - H_v^-1 g_v with the analytic gradient and observed Hessian of the summed NLL
     (estimator_r8.Objective.f_g / hess_vartheta, centred at the start point, Cholesky factorisation), no line search, no damping
     (DECISION 18-S2); before every step the Newton decrement eta = sqrt(g_v^T H_v^-1 g_v) is evaluated; the iteration stops as
     converged at the first iterate with eta <= 1e-5 (the round-8b threshold, inclusive as in estimator_r8.run_lbfgsb), and as not
     converged after 50 steps, at a non-finite objective / gradient / Hessian, or at a failed Cholesky factorisation;
  4. the converged iterate replaces the fallback theta_LS of that run in a supplementary copy of the CML estimates; a run that does
     not converge keeps theta_LS (DECISION 18-S3). Part B statistics (acc18.acc_stats, J from Part A) are recomputed on that copy.
"""
import os
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
import acc18 as a

e8, e8b = a.e8, a.e8b
P = a.P
MAX_STEPS = 50
TOL = e8.DECR_TOL_SUPP
assert TOL == 1e-5


def newton(data, theta0):
    """full Newton steps from theta0 (see module docstring, item 3)."""
    ob = e8.Objective(data, theta0)
    v = np.zeros(P); hist = []
    for step in range(MAX_STEPS+1):
        f, g = ob.f_g(v)
        Hs = ob.hess_vartheta(v)
        if not (np.isfinite(f) and np.all(np.isfinite(g)) and np.all(np.isfinite(Hs))):
            return dict(theta=ob.theta(v), converged=False, steps=step, reason='non-finite', hist=hist)
        try:
            cf = cho_factor(Hs, lower=True)
        except LinAlgError:
            return dict(theta=ob.theta(v), converged=False, steps=step, reason='cholesky', hist=hist)
        dv = cho_solve(cf, g); eta = float(np.sqrt(max(g@dv, 0.0)))
        hist.append(dict(step=step, f=float(f), eta=eta, grad_inf=float(np.abs(g).max())))
        if eta <= TOL:
            return dict(theta=ob.theta(v), converged=True, steps=step, reason='eta', hist=hist)
        if step == MAX_STEPS:
            break
        v = v-dv
    return dict(theta=ob.theta(v), converged=False, steps=MAX_STEPS, reason='max_steps', hist=hist)


def supp_one(args):
    """one non-accepted run: reproduce stage B's decision, then Newton from the first call's end point."""
    trig, i, ev, z, theta_LS_archived = args
    zt = np.where(ev, z, np.nan)
    d = a.stage_data(trig, ev, zt)
    res = e8b.estimate_stage1(d, 80261116+i)
    reproduced = bool((not res['converged']) and res['fallback'] == 'theta_LS'
                      and np.array_equal(np.asarray(res['theta'], float), theta_LS_archived)
                      and np.array_equal(np.asarray(res['theta_LS'], float), theta_LS_archived))
    a1, a2 = res['attempts']
    th1 = np.asarray(a1['theta'], float); th2 = np.asarray(a2['theta'], float)
    nt = newton(d, th1)
    return dict(trig=trig, i=int(i), N=int(ev.sum()), reproduced=reproduced,
                a1=dict(success=a1['success'], message=a1['message'], nit=a1['nit'], fun=a1['fun'], grad_inf=a1['grad_inf'],
                        decrement=a1['decrement']),
                a2=dict(success=a2['success'], message=a2['message'], nit=a2['nit'], fun=a2['fun'], grad_inf=a2['grad_inf'],
                        decrement=a2['decrement']),
                theta_a1=th1, theta_a2=th2, theta_LS=np.asarray(res['theta_LS'], float),
                newton=dict(theta=nt['theta'], converged=nt['converged'], steps=nt['steps'], reason=nt['reason'], hist=nt['hist']))
