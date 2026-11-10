"""Round 14: the one-parameter send-on-delta rule (prompt reports_CN/prompt_round14_one_parameter_rule_EN.md §1).

Continuous form (sensor):   lower edge = min(xbar_k - e sigma, r - sigma),  upper edge = max(xbar_k + e sigma, r + sigma),
                            transmit iff z_k <= lower edge or z_k >= upper edge.
Lattice form (evaluator):   on the sigma/24 design grid, array index p = 1 .. K-1 (step k = p + 1), node i:
                            L_i = max(24, i - FL[p] + n_p),   U_i = max(24, CL[p] - i + n_p)      (grid units, 24 = 1 sigma)
                            = eval11._rows for an edge action with offset n_p and minimum distance 24.
Mixed rule R(n, j), j = 0 .. 20: n_p = n where p mod 20 < j, n + 1 elsewhere; R(n, 20) = offset n everywhere, R(n, 0) = R(n + 1, 20).
Lattice parameter e_bar = (n + 1 - j/20)/24 sigma, mean physical offset e_eff = e_bar + 1/48 sigma.

The rule object has rows_full(p), so eval13.evaluate (sigma/24: eval13.forward with TOL24) and eval13.evaluate_refined(rule, 5) (sigma/120,
table semantics) evaluate it. One process per input (eval13 binds the input named by R13_INPUT at import).

Monte Carlo: _mc_lattice is round11/eval11._mc_batch with one change (the half-widths of the looked-up node follow the rule); _mc_cont simulates
the continuous form directly. Both use the same random-number order as eval11._mc_batch.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); R13 = os.path.join(EV, 'round13')
sys.path.insert(0, R13)
import numpy as np
from scipy.stats import norm
import eval13 as e13
import common13 as c13
ew, dr = e13.ew, e13.dr
s, K, G, B_NOM = e13.s, e13.K, e13.G, e13.B
Z, DZ, XK, HK, KA, KB, FL, CL, Z0 = e13.Z, e13.DZ, e13.XK, e13.HK, e13.KA, e13.KB, e13.FL, e13.CL, e13.Z0
MIN_HW = 24                       # 1 sigma minimum distance of the edges from r
PERIOD = 20                       # j/20 mixing of offsets n and n + 1
NAME = e13.NAME
Stop = c13.Stop


class Rule:
    """R(n, j): offset n_p = n at array indices p with p mod 20 < j, n + 1 elsewhere."""
    def __init__(self, n, j):
        n, j = int(n), int(j); assert 0 <= j <= PERIOD
        self.n, self.j = n, j
        pp = np.arange(K)
        self.n_p = np.where((pp % PERIOD) < j, n, n+1).astype(np.int64)
        self._i = np.arange(G, dtype=np.int64)
        self._L = np.empty(G, np.int64); self._U = np.empty(G, np.int64)
        self.X = e13.X_PAD
        self.e_bar = (n+1-j/PERIOD)/24.0; self.e_eff = self.e_bar+1/48.0
        self.n_mean = float(self.n_p[1:].mean())                  # exact mean offset over p = 1 .. K-1 (diagnostic)

    def rows_full(self, p, out=None):
        L, U = (self._L, self._U) if out is None else out
        npp = self.n_p[p]
        np.maximum(self._i-FL[p]+npp, MIN_HW, out=L)
        np.maximum(CL[p]-self._i+npp, MIN_HW, out=U)
        return L, U

    def is_sod(self): return False
    def label(self): return f"R({self.n}, {self.j})"


def evaluate(rule, f=1):
    """sigma/24 (f = 1, eval13.evaluate = eval13.forward with TOL24 + record) or sigma/(24 f) (eval13.evaluate_refined)."""
    t0 = time.perf_counter()
    if f == 1:
        rec, J, fw = e13.evaluate(rule); rec['G'] = G
    else:
        rec, J, fw = e13.evaluate_refined(rule, f)
    rec['wall_s'] = time.perf_counter()-t0; rec['f'] = f
    rec['cond'] = float(np.linalg.cond(J)); rec['rule'] = rule.label(); rec['n'] = rule.n; rec['j'] = rule.j
    rec['support_lo_min'] = int(fw['sup_lo'].min()); rec['support_hi_max'] = int(fw['sup_hi'].max())
    return rec, J


def fit_rule(Bt, log=print):
    """prompt §1 fitting procedure (deterministic, sigma/24):
      1. integer n with E[N](R(n, 20)) > Bt >= E[N](R(n, 0)) = E[N](R(n + 1, 20)); start n = round(24 Phi^-1(1 - q*)), q* = (Bt - 1)/(K - 1),
         step +-1;
      2. j0 = floor(20 (Bt - E(n + 1)) / (E(n) - E(n + 1))); evaluate R(n, j0); decrease j while infeasible; if R(n, j0) is feasible, increase
         j while j + 1 is feasible (DECISION 14-4); the j + 1 above the returned j is always evaluated and infeasible;
      3. the returned rule on sigma/120: E[N]_120 <= Bt required (else Stop)."""
    cache = {}
    def ev(n, j):
        key = (n+1, PERIOD) if j == 0 else (n, j)                  # R(n, 0) is R(n + 1, 20)
        if key not in cache:
            r = Rule(*key); rec, J = evaluate(r, 1); cache[key] = (rec, J)
            log(f"FIT B={Bt!r} {r.label()} e_bar={r.e_bar:.5f} sigma: sigma/24 EN {rec['EN']!r} S {rec['S']!r} logdet {rec['logdet']!r} "
                f"feasible {rec['EN'] <= Bt} mass_dev {rec['mass_dev_max']:.2e} ({rec['wall_s']:.1f} s)")
        return cache[key][0]['EN']
    q = (Bt-1.0)/(K-1.0); n0 = int(round(24*norm.isf(q))); n = n0; steps = []
    while True:
        a, b = ev(n, PERIOD), ev(n, 0)
        steps.append(n)
        if a > Bt >= b: break
        if b > Bt: n += 1                                           # even offset n + 1 sends too much: larger offsets
        elif a <= Bt: n -= 1
        if len(steps) > 40: raise Stop(f"fit_rule: no bracket for B = {Bt}")
    En, En1 = ev(n, PERIOD), ev(n, 0)
    j0 = int(np.floor(PERIOD*(Bt-En1)/(En-En1))); j0 = min(max(j0, 0), PERIOD-1)
    j = j0
    if ev(n, j) > Bt:
        while ev(n, j) > Bt: j -= 1                                  # j = 0 is feasible by step 1
    else:
        while j+1 < PERIOD and ev(n, j+1) <= Bt: j += 1
    ev(n, j+1)                                                       # confirm j + 1 infeasible (j + 1 = 20: pure n, cached)
    assert ev(n, j+1) > Bt and ev(n, j) <= Bt
    rule = Rule(n, j)
    key = (n+1, PERIOD) if j == 0 else (n, j)
    rec24, J24 = cache[key]
    rec120, J120 = evaluate(rule, 5)
    log(f"FIT B={Bt!r} -> {rule.label()} e_bar {rule.e_bar!r} e_eff {rule.e_eff!r}: sigma/24 EN {rec24['EN']!r} logdet {rec24['logdet']!r}; "
        f"sigma/120 EN {rec120['EN']!r} S {rec120['S']!r} logdet {rec120['logdet']!r} mass_dev {rec120['mass_dev_max']:.2e} ({rec120['wall_s']:.1f} s); "
        f"start n {n0}, n visited {steps}, j0 {j0}")
    res = dict(B=Bt, q_star=q, e1=float(norm.isf(q)), n_start=n0, n_steps=steps, n=n, j=j, j0=j0, e_bar=rule.e_bar, e_eff=rule.e_eff,
               n_mean_exact=rule.n_mean, delta=rule.e_eff-float(norm.isf(q)),
               evaluated24={f"R({k[0]}, {k[1]})": dict(EN=v[0]['EN'], S=v[0]['S'], logdet=v[0]['logdet'], feasible=bool(v[0]['EN'] <= Bt),
                                                       mass_dev_max=v[0]['mass_dev_max'], wall_s=v[0]['wall_s']) for k, v in sorted(cache.items())},
               rec24=rec24, rec120=rec120, j_plus_1_infeasible=bool(ev(n, j+1) > Bt), budget24=bool(rec24['EN'] <= Bt),
               budget120=bool(rec120['EN'] <= Bt))
    if not res['budget120']: raise Stop(f"fit_rule: sigma/120 E[N] {rec120['EN']!r} > B = {Bt!r} for {rule.label()}")
    return res, rule, J24, J120


# --- Monte Carlo ------------------------------------------------------------------------------------------------------------
def _mc_lattice(args):
    """round11/eval11._mc_batch with one change: half-widths of the looked-up node Lm = max(i - FL[p] + n_p, 24), Up = max(CL[p] - i + n_p, 24)."""
    n_p, R, seed = args
    z0 = Z0; dz = DZ
    rng = np.random.default_rng(seed)
    r = XK[0]+s*rng.standard_normal(R); N = np.ones(R); Wk = np.zeros(K); Wk[0] = float(R); S = np.ones(R)
    nclip = 0
    for p in range(1, K):
        i = np.ceil((r-z0)/dz-0.5).astype(np.int64)                     # nearest node, ties -> smaller index
        nclip += int(((i < 0) | (i > G-1)).sum()); i = np.clip(i, 0, G-1)
        Lm = np.maximum(i-FL[p]+n_p[p], MIN_HW); Up = np.maximum(CL[p]-i+n_p[p], MIN_HW)
        lo = r-Lm*dz; hi = r+Up*dz
        cev, cno = dr.cfun((lo-XK[p])/s, (hi-XK[p])/s); c = cev+cno
        Wk[p] = c.sum(); S += c
        zk = XK[p]+s*rng.standard_normal(R)
        snd = (zk <= lo) | (zk >= hi)
        N += snd; r = np.where(snd, zk, r)
    return Wk, N, S, nclip

def _mc_cont(args):
    """the continuous rule: lo = min(XK[p] - e s, r - s), hi = max(XK[p] + e s, r + s); same random-number order as _mc_lattice."""
    e, R, seed = args
    rng = np.random.default_rng(seed)
    r = XK[0]+s*rng.standard_normal(R); N = np.ones(R); Wk = np.zeros(K); Wk[0] = float(R); S = np.ones(R)
    for p in range(1, K):
        lo = np.minimum(XK[p]-e*s, r-s); hi = np.maximum(XK[p]+e*s, r+s)
        cev, cno = dr.cfun((lo-XK[p])/s, (hi-XK[p])/s); c = cev+cno
        Wk[p] = c.sum(); S += c
        zk = XK[p]+s*rng.standard_normal(R)
        snd = (zk <= lo) | (zk >= hi)
        N += snd; r = np.where(snd, zk, r)
    return Wk, N, S, 0

def monte_carlo(kind, par, R=10000, batches=20, seed=20261109, workers=4):
    """aggregation as round11/eval11.monte_carlo: E[N], S with standard errors of the mean, log det J of the mean weights, batch standard
    error of log det J."""
    from multiprocessing import Pool
    fn = _mc_lattice if kind == 'lattice' else _mc_cont
    jobs = [(par, R//batches, seed+b) for b in range(batches)]
    t0 = time.perf_counter()
    with Pool(workers) as pool: outs = pool.map(fn, jobs)
    Ns = np.concatenate([x[1] for x in outs]); Ss = np.concatenate([x[2] for x in outs]); Wk = sum(x[0] for x in outs)/R
    J = (HK*Wk[:, None]).T@HK/s**2; ld = float(np.linalg.slogdet(J)[1])
    ldb = np.array([np.linalg.slogdet((HK*(x[0]/(R//batches))[:, None]).T@HK/s**2)[1] for x in outs])
    return dict(kind=kind, R=R, batches=batches, seed=seed, EN_mc=float(Ns.mean()), EN_se=float(Ns.std(ddof=1)/np.sqrt(R)), S_mc=float(Ss.mean()),
                S_se=float(Ss.std(ddof=1)/np.sqrt(R)), logdet_mc=ld, logdet_batch_se=float(ldb.std(ddof=1)/np.sqrt(batches)),
                cond_mc=float(np.linalg.cond(J)), n_clipped=int(sum(x[3] for x in outs)), wall_s=time.perf_counter()-t0)
