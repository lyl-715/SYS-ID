"""Round 8b: the estimator with the round-8b convergence acceptance (spec reports_CN/plan_round8b_CN.md v2, frozen; the only
algorithmic change against round 8 v3.1 §1.2 / §1.5).

Everything is round 8's estimator_r8 (imported, not copied): the likelihood and its stable log window probabilities, the
SUMMED objective, the centred arithmetic, the analytic gradient and Hessian, L-BFGS-B with gtol = 1e-8, ftol = 0,
maxiter = 500 and the scaling s = [2, 1, ..., 1], the LS initial value / theta_bk, the restart from start + delta with
delta = 0.01 s * d, d = default_rng(seed).standard_normal(9), and the fallbacks. The only differences:

  acceptance  an optimiser call is accepted iff result.success is True; parameters, objective and gradient are all finite;
              the Cholesky factorisation of the analytic Hessian H_vartheta of the summed NLL succeeds; and the Newton
              decrement eta = sqrt(g^T H_vartheta^-1 g) <= 1e-5. This is exactly the rule round 8's run_lbfgsb evaluates as
              accept['supp'] (the supplementary analysis S of dcc660b, kept exactly). Any termination with success False
              (line-search failure included) is not accepted however small eta is. No extra Newton step.
  seeds       the restart seed is an explicit argument (official run: 60261028 + r for stage 1, 70261028 + r for the joint
              estimates; the A3 checks and the reproduction of S pass round-8 seeds).

As in round 8 (DECISION E-4) both optimiser calls are always run; the restart's result is used only if the first call is not
accepted.

Imports: numpy, declared, estimator_r8 (round 8; numpy, scipy, declared, tables_r8) -- nothing that holds theta_0.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); R8 = os.path.join(os.path.dirname(HERE), 'round8')
for _p in (HERE, R8):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np
import declared as dk
import estimator_r8 as e8

RULE_KEY = 'supp'                         # round 8's key of the Newton-decrement acceptance (estimator_r8.run_lbfgsb)
RULE_NAME = 'newton_decrement'
DECR_TOL = e8.DECR_TOL_SUPP
assert DECR_TOL == 1e-5 and (e8.GTOL, e8.FTOL, e8.MAXITER) == (1e-8, 0.0, 500) and np.array_equal(e8.SC, np.r_[2.0, np.ones(8)])

StageData, stack, nll_grad_hess_theta, Objective = e8.StageData, e8.stack, e8.nll_grad_hess_theta, e8.Objective
attempt_summary = e8._attempt_summary


def accepted(att):
    """the round-8b acceptance of one optimiser call (estimator_r8.run_lbfgsb output). The independent re-evaluation from the
    archived fields must agree with run_lbfgsb's own flag: eta is finite iff H_vartheta was finite and its Cholesky factor
    succeeded (run_lbfgsb leaves eta = inf otherwise)."""
    flag = bool(att['accept'][RULE_KEY])
    again = bool(att['success'] and att['finite'] and np.isfinite(att['decrement']) and att['decrement'] <= DECR_TOL)
    assert flag == again, ('acceptance flags disagree', flag, again)
    return flag

def decide(att1, att2, fallback, fallback_kind):
    """(theta, converged, used, fallback flag) under the round-8b acceptance (estimator_r8.decide with the 'supp' key)."""
    accepted(att1); accepted(att2)
    return e8.decide(att1, att2, RULE_KEY, fallback, fallback_kind)

def estimate_stage1(stage1, restart_seed):
    """spec §1.2 (round 8 v3.1) with the round-8b acceptance: LS initial value if usable (rank 9, finite) else theta_bk;
    first call from theta_init, restart from theta_init + delta(restart_seed); fallback theta_LS if usable else theta_bk."""
    th_ls, usable, rank = e8.ls_init(stage1)
    th_init = th_ls if usable else dk.THETA_BK.copy()
    a1 = e8.run_lbfgsb(stage1, th_init)
    a2 = e8.run_lbfgsb(stage1, th_init+e8.restart_delta(restart_seed))
    fb, fbk = (th_ls, 'theta_LS') if usable else (dk.THETA_BK.copy(), 'theta_bk')
    th, conv, used, fk = decide(a1, a2, fb, fbk)
    return dict(theta_LS=th_ls, LS_usable=usable, LS_rank=rank, theta_init=th_init, init_kind='theta_LS' if usable else 'theta_bk',
                restart_seed=int(restart_seed), attempts=[attempt_summary(a1), attempt_summary(a2)],
                rule=RULE_NAME, theta=th, converged=conv, used=used, fallback=fk)

def estimate_joint(stages, theta_hat1, restart_seed):
    """spec §1.5: l_1 + l_2 with T fixed inside `stages`, first call from theta_hat_1, restart from theta_hat_1 +
    delta(restart_seed), fallback theta_hat_1."""
    data = e8.stack(stages)
    a1 = e8.run_lbfgsb(data, theta_hat1)
    a2 = e8.run_lbfgsb(data, np.asarray(theta_hat1, float)+e8.restart_delta(restart_seed))
    th, conv, used, fk = decide(a1, a2, theta_hat1, 'theta_hat_1')
    return dict(theta_final=th, converged=conv, used=used, fallback=fk, restart_seed=int(restart_seed),
                attempts=[attempt_summary(a1), attempt_summary(a2)])
