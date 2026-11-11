"""Round 17 (prompt reports_CN/prompt_round17_absolute_diagnostics_EN.md): diagnostics of the information matrix in absolute terms (Part A) and
the one-parameter rule with a minimum half-width M_min (Part B). Input U only (R13_INPUT must be U). Rounds 4-16 are imported read-only.

Diagnostics of J against J_full = HK^T HK / s^2 (diag): I = log det J; generalised eigenvalues mu_1 <= ... <= mu_9 of J v = mu J_full v
(scipy.linalg.eigh(J, J_full)), with the check exp(mean log mu) = exp((I - I_full)/9); cond(J) (2-norm, numpy.linalg.cond), lambda_min,
lambda_max; per-parameter Cramer-Rao standard-deviation ratios sqrt(diag(J^-1)/diag(J_full^-1)) and their geometric mean exp((I_full - I)/18);
tr(J^-1)/tr(J_full^-1) and lambda_min(J)/lambda_min(J_full).

Rule with minimum half-width M_min (grid units): L_i = max(M_min, i - FL[p] + n_p), U_i = max(M_min, CL[p] - i + n_p); R(n, j) mixes offsets n
(p mod 20 < j) and n + 1; M_min = 24 is Round 14's rule14.Rule. fit_rule17 is Round 14's fit_rule with this rule. The continuous rule:
lo = min(x - e s, r - m s), hi = max(x + e s, r + m s), m = M_min/24 (Round 14's _mc_cont with the r-side distance m).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
for _p in (os.path.join(EV, 'round13'), os.path.join(EV, 'round14')): sys.path.insert(0, _p)
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.stats import norm
import eval13 as e13
import rule14 as r14                                  # round 14 (read-only): evaluate(), Rule (M_min = 24)
ew, dr = e13.ew, e13.dr
assert e13.NAME == 'U', "round 17 is for input U only (R13_INPUT=U)"
s, K, G, B = e13.s, e13.K, e13.G, e13.B
Z, DZ, XK, HK, KA, KB, FL, CL, Z0 = e13.Z, e13.DZ, e13.XK, e13.HK, e13.KA, e13.KB, e13.FL, e13.CL, e13.Z0
J_FULL = HK.T@HK/s**2
PERIOD = 20
PAR_NAMES = ('G(0)', 'Re G(jw1)', 'Im G(jw1)', 'Re G(jw2)', 'Im G(jw2)', 'Re G(jw3)', 'Im G(jw3)', 'Re G(jw4)', 'Im G(jw4)')


# --- diagnostics of J -------------------------------------------------------------------------------------------------------------
def inv_spd(J):
    c = cho_factor(J, lower=True)
    return cho_solve(c, np.eye(J.shape[0]))

def diagnostics(J, Jf=J_FULL):
    J = np.asarray(J, float)
    sg, I = np.linalg.slogdet(J); sgf, If = np.linalg.slogdet(Jf)
    assert sg > 0 and sgf > 0
    mu = eigh(J, Jf, eigvals_only=True)
    geo = float(np.exp(np.mean(np.log(mu)))); geo_ref = float(np.exp((I-If)/9))
    lam = np.linalg.eigvalsh(J); lamf = np.linalg.eigvalsh(Jf)
    Ji = inv_spd(J); Jfi = inv_spd(Jf)
    sd = np.sqrt(np.diag(Ji)/np.diag(Jfi))
    return dict(I=float(I), I_minus_full=float(I-If), mu=[float(v) for v in mu], mu_min=float(mu[0]), mu_max=float(mu[-1]), mu_ratio=float(mu[-1]/mu[0]),
                geo_mu=geo, geo_ref=geo_ref, geo_check_rel=abs(geo-geo_ref)/geo_ref, cond=float(np.linalg.cond(J)), lam_min=float(lam[0]), lam_max=float(lam[-1]),
                sd_ratio=[float(v) for v in sd], sd_min=float(sd.min()), sd_max=float(sd.max()), sd_geo=float(np.exp((If-I)/18)),
                sd_geo_check=float(np.exp(np.mean(np.log(sd)))), tr_ratio=float(np.trace(Ji)/np.trace(Jfi)), lam_min_ratio=float(lam[0]/lamf[0]),
                lam_max_ratio=float(lam[-1]/lamf[-1]))


# --- rule with a minimum half-width M_min -----------------------------------------------------------------------------------------
class Rule17:
    def __init__(self, n, j, M_min):
        n, j, M_min = int(n), int(j), int(M_min); assert 0 <= j <= PERIOD and M_min >= 1
        self.n, self.j, self.M_min = n, j, M_min
        pp = np.arange(K); self.n_p = np.where((pp % PERIOD) < j, n, n+1).astype(np.int64)
        self._i = np.arange(G, dtype=np.int64); self._L = np.empty(G, np.int64); self._U = np.empty(G, np.int64)
        self.X = e13.X_PAD
        self.e_bar = (n+1-j/PERIOD)/24.0; self.e_eff = self.e_bar+1/48.0

    def rows_full(self, p, out=None):
        L, U = (self._L, self._U) if out is None else out
        npp = self.n_p[p]
        np.maximum(self._i-FL[p]+npp, self.M_min, out=L)
        np.maximum(CL[p]-self._i+npp, self.M_min, out=U)
        return L, U

    def is_sod(self): return False
    def label(self): return f"R({self.n}, {self.j}; M_min {self.M_min})"


def fit_rule17(Bt, M_min, log=print):
    """Round 14's fit_rule (rule14.fit_rule) with Rule17(n, j, M_min): start n = round(24 Phi^-1(1 - q*)), step +-1 until
    E(R(n, 20)) > Bt >= E(R(n + 1, 20)); j0 = floor(20 (Bt - E(n + 1))/(E(n) - E(n + 1))); decrease j while infeasible, else increase while
    j + 1 is feasible; confirm j + 1 infeasible; sigma/120 E[N] <= Bt required."""
    cache = {}
    def ev(n, j):
        key = (n+1, PERIOD) if j == 0 else (n, j)
        if key not in cache:
            r = Rule17(*key, M_min); rec, J = r14.evaluate(r, 1); cache[key] = (rec, J)
            log(f"FIT M_min={M_min} {r.label()} e_bar={r.e_bar:.5f}: sigma/24 EN {rec['EN']!r} S {rec['S']!r} logdet {rec['logdet']!r} feasible {rec['EN'] <= Bt} "
                f"mass_dev {rec['mass_dev_max']:.2e} ({rec['wall_s']:.1f} s)")
        return cache[key][0]['EN']
    q = (Bt-1.0)/(K-1.0); n0 = int(round(24*norm.isf(q))); n = n0; steps = []
    while True:
        a, b = ev(n, PERIOD), ev(n, 0); steps.append(n)
        if a > Bt >= b: break
        if b > Bt: n += 1
        elif a <= Bt: n -= 1
        if len(steps) > 40: raise RuntimeError(f"fit_rule17: no bracket (M_min {M_min})")
    En, En1 = ev(n, PERIOD), ev(n, 0)
    j0 = int(np.floor(PERIOD*(Bt-En1)/(En-En1))); j0 = min(max(j0, 0), PERIOD-1); j = j0
    if ev(n, j) > Bt:
        while ev(n, j) > Bt: j -= 1
    else:
        while j+1 < PERIOD and ev(n, j+1) <= Bt: j += 1
    ev(n, j+1)
    assert ev(n, j+1) > Bt and ev(n, j) <= Bt
    rule = Rule17(n, j, M_min)
    key = (n+1, PERIOD) if j == 0 else (n, j)
    rec24, J24 = cache[key]
    rec120, J120 = r14.evaluate(rule, 5)
    log(f"FIT M_min={M_min} -> {rule.label()} e_eff {rule.e_eff!r}: sigma/24 EN {rec24['EN']!r} S {rec24['S']!r} logdet {rec24['logdet']!r}; sigma/120 EN {rec120['EN']!r} "
        f"S {rec120['S']!r} logdet {rec120['logdet']!r} mass_dev {rec120['mass_dev_max']:.2e} ({rec120['wall_s']:.1f} s); start n {n0}, visited {steps}, j0 {j0}")
    res = dict(M_min=M_min, m_sigma=M_min/24, B=Bt, q_star=q, n_start=n0, n_steps=steps, n=n, j=j, j0=j0, e_bar=rule.e_bar, e_eff=rule.e_eff,
               evaluated24={f"R({k[0]}, {k[1]})": dict(EN=v[0]['EN'], S=v[0]['S'], logdet=v[0]['logdet'], feasible=bool(v[0]['EN'] <= Bt),
                                                       mass_dev_max=v[0]['mass_dev_max'], wall_s=v[0]['wall_s']) for k, v in sorted(cache.items())},
               j_plus_1_infeasible=bool(ev(n, j+1) > Bt), budget24=bool(rec24['EN'] <= Bt), budget120=bool(rec120['EN'] <= Bt))
    if not res['budget120']: raise RuntimeError(f"fit_rule17: sigma/120 E[N] {rec120['EN']!r} > B (M_min {M_min})")
    return res, rule, rec24, J24, rec120, J120


# --- continuous rule with r-side distance m -----------------------------------------------------------------------------------------
def _mc_cont_m(args):
    """Round 14's _mc_cont (rule14._mc_cont) with lo = min(x - e s, r - m s), hi = max(x + e s, r + m s); same random-number order."""
    e, m, R, seed = args
    rng = np.random.default_rng(seed)
    r = XK[0]+s*rng.standard_normal(R); N = np.ones(R); Wk = np.zeros(K); Wk[0] = float(R); S = np.ones(R)
    for p in range(1, K):
        lo = np.minimum(XK[p]-e*s, r-m*s); hi = np.maximum(XK[p]+e*s, r+m*s)
        cev, cno = dr.cfun((lo-XK[p])/s, (hi-XK[p])/s); c = cev+cno
        Wk[p] = c.sum(); S += c
        zk = XK[p]+s*rng.standard_normal(R)
        snd = (zk <= lo) | (zk >= hi)
        N += snd; r = np.where(snd, zk, r)
    return Wk, N, S

def mc_cont(e, m, R=10000, batches=20, seed=20261109, workers=4):
    """aggregation as rule14.monte_carlo (Round 14 seeds 20261109 + b, 20 batches)."""
    from multiprocessing import Pool
    jobs = [(e, m, R//batches, seed+b) for b in range(batches)]
    t0 = time.perf_counter()
    with Pool(workers) as pool: outs = pool.map(_mc_cont_m, jobs)
    Ns = np.concatenate([x[1] for x in outs]); Ss = np.concatenate([x[2] for x in outs]); Wk = sum(x[0] for x in outs)/R
    J = (HK*Wk[:, None]).T@HK/s**2; ld = float(np.linalg.slogdet(J)[1])
    ldb = np.array([np.linalg.slogdet((HK*(x[0]/(R//batches))[:, None]).T@HK/s**2)[1] for x in outs])
    return dict(e=e, m=m, R=R, EN_mc=float(Ns.mean()), EN_se=float(Ns.std(ddof=1)/np.sqrt(R)), S_mc=float(Ss.mean()), S_se=float(Ss.std(ddof=1)/np.sqrt(R)),
                logdet_mc=ld, logdet_batch_se=float(ldb.std(ddof=1)/np.sqrt(batches)), wall_s=time.perf_counter()-t0), J
