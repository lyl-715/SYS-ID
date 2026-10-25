"""Round 8, Part A1: the frozen ML estimator (spec reports_CN/plan_round8_CN.md v3.1, frozen; §1.2, §1.5, §1.7).

Likelihood (the §8.4 censored-Gaussian likelihood of mc84.py, extended to general (k, r)-dependent windows): a transmitted
sample contributes the Gaussian density phi((z_k - h_k^T theta)/sigma)/sigma, a non-transmitted step the window probability
Phi(beta_k) - Phi(alpha_k), alpha_k = (lo_k - h_k^T theta)/sigma, beta_k = (up_k - h_k^T theta)/sigma, where the window
(lo_k, up_k) = (r_k - M^-_k h, r_k + M^+_k h) is given by the stage's trigger rule with r_k reconstructed from the record (the
last transmitted value before k): constant Delta* (M^- h = M^+ h = Delta*) for an SoD stage, the fixed table T (looked up from
r_k, tables_r8.SeqTable) for the sequential stage 2. T is data here: it is never redesigned inside the optimisation.

Objective: the SUMMED negative log-likelihood (no division by the number of observations; the theta-independent constants
log sigma + log(2 pi)/2 per transmitted sample are dropped, as in mc84.py). log P_k by the stable forms of mc84.py
(log_ndtr and log1p(-exp(.)), lower-tail form for alpha + beta < 0, upper-tail form otherwise); phi(.)/P via exp of logs.
Gradient and observed Hessian analytic (mc84.py expressions; the Hessian is used for the Newton decrement of the supplementary
acceptance rule and for the Hessian-FIM diagnostic H_r = -grad^2 l_2(theta_0)).

Arithmetic (DECISION E-2): every optimiser call evaluates the objective in a centred form about its own start point theta_ref:
residuals (z_k - h_k^T theta_ref)/sigma and window ends (lo_k - h_k^T theta_ref)/sigma are formed once, and theta = theta_ref
+ s * v with the optimisation variable v = vartheta - vartheta_ref. This is the same objective and the same optimiser
(L-BFGS-B is translation invariant), only the floating-point representation of the parameter offset is finer.

Optimiser (frozen): scipy L-BFGS-B in vartheta = theta / s, s = [2, 1, ..., 1], analytic gradient, gtol = 1e-8, ftol = 0,
maxiter = 500 (other options scipy defaults). Acceptance rules:
  'frozen' (spec §1.2, primary): result.success, parameters / objective / gradient finite, ||grad_vartheta l||_inf <= 1e-8.
  'supp'   (supplementary analysis, NOT the frozen rule; DECISION E-1): identical except the last condition is replaced by the
           Newton decrement in vartheta, sqrt(g^T Hs^-1 g) <= 1e-5 (Hs = observed Hessian in vartheta, Cholesky; failure ->
           not accepted), i.e. the remaining optimisation error is below 1e-5 statistical standard errors.
Initial value (stage 1): least squares on the transmitted samples treated as complete observations, usable iff the
transmitted-sample design matrix has rank 9 (numpy matrix_rank, default tolerance) and the result is finite; otherwise
theta_bk. Restart (once): from theta_init + delta, delta = 0.01 s * d, d = default_rng(seed).standard_normal(9), seed
30261028 + r (stage 1) / 40261028 + r (joint). Fallbacks: stage 1 -> theta_LS if usable else theta_bk; joint -> theta_hat_1.

Imports: numpy, scipy, declared, tables_r8 -- nothing that holds theta_0.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import numpy as np
from scipy.special import log_ndtr
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, LinAlgError
import declared as dk

s = dk.s; D = dk.D; K = dk.K; SC = dk.SCALE; NP = dk.N_PAR
L2PI = 0.5*np.log(2*np.pi)
GTOL, FTOL, MAXITER = 1e-8, 0.0, 500
GRAD_TOL_FROZEN = 1e-8
DECR_TOL_SUPP = 1e-5
SEED_R1, SEED_RJ = 30261028, 40261028
RULES = ('frozen', 'supp')


# --- records -> likelihood data ------------------------------------------------------------------------------------------
def references(ev, zt):
    """r_k reconstructed from the record: the last transmitted value before step k (r[0] = nan; step 1 is transmitted)."""
    ev = np.asarray(ev, bool); assert ev[0]
    idx = np.where(ev, np.arange(len(ev)), 0); np.maximum.accumulate(idx, out=idx)
    r = np.empty(len(ev)); r[0] = np.nan; r[1:] = zt[idx[:-1]]
    return r

class StageData:
    """one stage's record with its trigger rule: rule = ('sod',) (constant Delta*) or ('table', SeqTable)."""
    def __init__(self, ev, zt, rule):
        Hm = dk.H(); ev = np.asarray(ev, bool); zt = np.asarray(zt, float)
        r = references(ev, zt); ne = ~ev; kk = np.flatnonzero(ne)
        self.ev = ev; self.N = int(ev.sum())
        self.He = Hm[ev]; self.ze = zt[ev]; self.Hn = Hm[ne]; self.rn = r[ne]
        if rule[0] == 'sod':
            self.mm = np.full(kk.size, D); self.mp = np.full(kk.size, D)
        else:
            tab = rule[1]; Mm, Mp = tab.lookup_vec(kk+1, self.rn)          # step k = index + 1
            self.mm = Mm*tab.h; self.mp = Mp*tab.h; self.Mm, self.Mp = Mm, Mp
        self.lo = self.rn-self.mm; self.up = self.rn+self.mp
        assert np.all(np.isfinite(self.ze)) and np.all(np.isfinite(self.lo)) and np.all(np.isfinite(self.up))

def stack(stages):
    """concatenate several stages (the joint likelihood l_1 + l_2 is the sum over both records)."""
    o = StageData.__new__(StageData)
    o.He = np.vstack([q.He for q in stages]); o.ze = np.concatenate([q.ze for q in stages])
    o.Hn = np.vstack([q.Hn for q in stages]); o.lo = np.concatenate([q.lo for q in stages]); o.up = np.concatenate([q.up for q in stages])
    o.N = sum(q.N for q in stages)
    return o


# --- the objective (mc84.py expressions, general windows, centred arithmetic) ---------------------------------------------
def logP(al, be):
    """log(Phi(be) - Phi(al)) by the stable forms of mc84.py (unused branch may warn, the selected one is finite)."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        lo = log_ndtr(be)+np.log1p(-np.exp(log_ndtr(al)-log_ndtr(be)))
        up = log_ndtr(-al)+np.log1p(-np.exp(log_ndtr(-be)-log_ndtr(-al)))
    return np.where(al+be < 0, lo, up)

class Objective:
    """summed negative log-likelihood of `data` in the variable v (vartheta - vartheta_ref), centred at theta_ref."""
    def __init__(self, data, theta_ref):
        self.d = data; self.ref = np.array(theta_ref, float)
        mu_e = data.He@self.ref; mu_n = data.Hn@self.ref
        self.de = (data.ze-mu_e)/s; self.al0 = (data.lo-mu_n)/s; self.be0 = (data.up-mu_n)/s
        self.nfev = 0

    def theta(self, v): return self.ref+SC*np.asarray(v, float)

    def _parts(self, v):
        dth = SC*np.asarray(v, float)
        res = self.de-(self.d.He@dth)/s; an = (self.d.Hn@dth)/s
        al = self.al0-an; be = self.be0-an
        lp = logP(al, be)
        return res, al, be, lp

    def f_g(self, v):
        """objective and gradient in vartheta."""
        self.nfev += 1
        res, al, be, lp = self._parts(v)
        f = 0.5*np.sum(res*res)-np.sum(lp)
        ea = np.exp(-0.5*al*al-L2PI-lp); eb = np.exp(-0.5*be*be-L2PI-lp)       # phi(al)/P, phi(be)/P
        g = self.d.He.T@(-res/s)+self.d.Hn.T@(-(ea-eb)/s)                      # gradient in theta
        return float(f), SC*g

    def hess_theta(self, v):
        """observed information -grad^2 log L in theta (mc84.py: transmitted 1/sigma^2, window ((be eb - al ea) + (ea - eb)^2)/sigma^2)."""
        res, al, be, lp = self._parts(v)
        ea = np.exp(-0.5*al*al-L2PI-lp); eb = np.exp(-0.5*be*be-L2PI-lp)
        c = (be*eb-al*ea)+(ea-eb)**2
        return (self.d.He.T@self.d.He+(self.d.Hn*c[:, None]).T@self.d.Hn)/s**2

    def hess_vartheta(self, v): return SC[:, None]*self.hess_theta(v)*SC[None, :]


def nll_grad_hess_theta(data, theta):
    """(f, gradient in theta, Hessian in theta) at theta, centred at theta itself (diagnostics, H_r)."""
    ob = Objective(data, theta); f, gv = ob.f_g(np.zeros(NP))
    return f, gv/SC, ob.hess_theta(np.zeros(NP))


# --- one optimiser call and the acceptance rules ---------------------------------------------------------------------------
def run_lbfgsb(data, theta_start):
    ob = Objective(data, theta_start)
    res = minimize(ob.f_g, np.zeros(NP), jac=True, method='L-BFGS-B', options=dict(gtol=GTOL, ftol=FTOL, maxiter=MAXITER))
    th = ob.theta(res.x); g = np.asarray(res.jac, float); f = float(res.fun)
    finite = bool(np.all(np.isfinite(res.x)) and np.all(np.isfinite(th)) and np.isfinite(f) and np.all(np.isfinite(g)))
    ginf = float(np.max(np.abs(g))) if finite else float('inf')
    decr = float('inf'); chol_ok = False
    if finite:
        try:
            Hs = ob.hess_vartheta(res.x)
            if np.all(np.isfinite(Hs)):
                cf = cho_factor(Hs, lower=True); decr = float(np.sqrt(max(g@cho_solve(cf, g), 0.0))); chol_ok = True
        except LinAlgError:
            pass
    acc = dict(frozen=bool(res.success and finite and ginf <= GRAD_TOL_FROZEN),
               supp=bool(res.success and finite and chol_ok and decr <= DECR_TOL_SUPP))
    return dict(theta=th, theta_start=np.array(theta_start, float), success=bool(res.success), message=str(res.message),
                status=int(res.status), nit=int(res.nit), nfev=int(res.nfev), fun=f, grad_inf=ginf,
                grad_2=float(np.linalg.norm(g)) if finite else float('inf'), decrement=decr, finite=finite, accept=acc)

def restart_delta(seed):
    d = np.random.default_rng(seed).standard_normal(NP)
    return 0.01*SC*d

def ls_init(stage1):
    """least squares on the transmitted samples treated as complete observations; usable iff rank 9 and finite."""
    He, ze = stage1.He, stage1.ze
    rank = int(np.linalg.matrix_rank(He))
    th = np.linalg.lstsq(He, ze, rcond=None)[0] if He.shape[0] > 0 else np.full(NP, np.nan)
    usable = bool(rank == NP and np.all(np.isfinite(th)))
    return th, usable, rank

def _attempt_summary(a):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in a.items()}

def decide(att1, att2, rule, fallback, fallback_kind):
    """apply one acceptance rule to (first call, restart call): returns (theta, converged, used, fallback flag)."""
    if att1['accept'][rule]: return att1['theta'], True, 'first', None
    if att2['accept'][rule]: return att2['theta'], True, 'restart', None
    return np.array(fallback, float), False, 'none', fallback_kind

def estimate_stage1(stage1, r):
    """spec §1.2 for replication r: both optimiser calls are run (the restart's result is used only when the first call is
    not accepted under a rule); returns the per-rule decision and the full diagnostics."""
    th_ls, usable, rank = ls_init(stage1)
    th_init = th_ls if usable else dk.THETA_BK.copy()
    a1 = run_lbfgsb(stage1, th_init)
    a2 = run_lbfgsb(stage1, th_init+restart_delta(SEED_R1+r))
    fb, fbk = (th_ls, 'theta_LS') if usable else (dk.THETA_BK.copy(), 'theta_bk')
    out = dict(theta_LS=th_ls, LS_usable=usable, LS_rank=rank, theta_init=th_init, init_kind='theta_LS' if usable else 'theta_bk',
               restart_seed=SEED_R1+r, attempts=[_attempt_summary(a1), _attempt_summary(a2)], rules={})
    for rule in RULES:
        th, conv, used, fk = decide(a1, a2, rule, fb, fbk)
        out['rules'][rule] = dict(theta=th, converged=conv, used=used, fallback=fk)
    return out

def estimate_joint(stages, theta_hat1, r):
    """spec §1.5: joint estimate from theta_hat_1 (T fixed inside `stages`); restart seed 40261028 + r; fallback theta_hat_1.
    The acceptance rule is applied by the caller via decide() (the start point theta_hat_1 differs between rules)."""
    data = stack(stages)
    a1 = run_lbfgsb(data, theta_hat1)
    a2 = run_lbfgsb(data, np.asarray(theta_hat1, float)+restart_delta(SEED_RJ+r))
    return a1, a2
