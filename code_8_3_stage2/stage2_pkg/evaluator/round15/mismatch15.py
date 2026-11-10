"""Round 15: the one-parameter rule of Round 14 when the sensor's model is an estimate (prompt
reports_CN/prompt_round15_rule_with_estimated_model_EN.md). Input U only (R13_INPUT must be U).

Models (§1): theta~ = theta* + delta. theta* = [G(0), Re G(j w_1), Im G(j w_1), ..., Im G(j w_4)] (order of dr.regressors; HK @ theta* = XK
to rounding). Estimated models delta = sqrt(c) solve(Lc^T, xi), Lc = cholesky(J_SoD) (lower), J_SoD = round4/results/J_U_SoD.npy:
main set (seed 20261112, c in (1, 2, 4), 40 draws each), reference set (seed 20261111, 4 draws each); scaled models (the 40 main-set c = 1
directions rescaled to a largest prediction error of eps sigma, eps in (0.5, 1, 2, 4)); gross error delta = +-0.02 theta*.

Continuous rule (§2): prediction xbar_k = x_k + h_k^T delta (DECISION 15-2: the true output XK plus the prediction error, so delta = 0 is the
true model exactly); lo = min(xbar_k - e_k s, r - s), hi = max(xbar_k + e_k s, r + s); transmit iff z_k <= lo or z_k >= hi; information
fraction c = dr.cfun((lo - x_k)/s, (hi - x_k)/s) with the TRUE output x_k. Variants: V0 (e_k = e_0), VM(Delta) (e_0 + Delta),
VF(T, m) (e_k = clip(e_0 + (N_k - (1 + q_t (k - 1)))/(2 q_t T), 0.5, 4.0), q_t = (B - m - 1)/(K - 1), per trajectory).

Monte Carlo (§3): vectorised over trajectories as eval11._mc_batch (r = x_1 + s noise, then per step the measurement noise), one job = one
(model, variant) with R trajectories and one seed.

Mismatched lattice rule (§5 checks 1-2): Round 14's R(37, 13) with FL, CL replaced by those of the prediction, FLd = floor((xbar - Z0)/DZ),
CLd = ceil(...); evaluated with the true XK (kernel windows and information) by eval13, and simulated by Round 14's routine 1c with the same
replacement.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R13 = os.path.join(EV, 'round13'); R4 = os.path.join(EV, 'round4')
sys.path.insert(0, R13)
import numpy as np
import eval13 as e13
ew, dr = e13.ew, e13.dr
assert e13.NAME == 'U', "round 15 is for input U only (R13_INPUT=U)"
s, K, G, B = e13.s, e13.K, e13.G, e13.B
Z, DZ, XK, HK, KA, KB, Z0 = e13.Z, e13.DZ, e13.XK, e13.HK, e13.KA, e13.KB, e13.Z0
I_SOD = 197.20097668224076
J_SOD = np.load(os.path.join(R4, 'results', 'J_U_SoD.npy'))
LEVELS = (1.0, 2.0, 4.0)
EPS = (0.5, 1.0, 2.0, 4.0)
MARGINS = (0.05, 0.10, 0.15, 0.20, 0.30)
FB_T = (50, 100, 300, 1000); FB_M = (0, 3, 6)
E_LO, E_HI = 0.5, 4.0
MIN_HW = 24


# --- models ---------------------------------------------------------------------------------------------------------------------
def theta_star():
    th = np.empty(9); th[0] = dr.G(0); th[1::2] = dr.Gj.real; th[2::2] = dr.Gj.imag
    return th

def pred_error(delta):
    d = HK@np.asarray(delta, float)/s
    return float(np.sqrt(np.mean(d*d))), float(np.abs(d).max())

def models():
    """name -> dict(set, c, d, eps, sign, delta (list), rms, max)."""
    Lc = np.linalg.cholesky(J_SOD)
    M = {}
    def add(name, delta, **kw):
        rms, mx = pred_error(delta); M[name] = dict(kw, delta=[float(v) for v in delta], rms=rms, max=mx)
    rng = np.random.default_rng(20261112)
    for c in LEVELS:
        for d in range(40):
            xi = rng.standard_normal(9); add(f'main_c{c:g}_d{d:02d}', np.sqrt(c)*np.linalg.solve(Lc.T, xi), set='main', c=c, d=d)
    rng = np.random.default_rng(20261111)
    for c in LEVELS:
        for d in range(4):
            xi = rng.standard_normal(9); add(f'ref_c{c:g}_d{d}', np.sqrt(c)*np.linalg.solve(Lc.T, xi), set='ref', c=c, d=d)
    for d in range(40):
        base = np.array(M[f'main_c1_d{d:02d}']['delta']); mx = float(np.abs(HK@base).max())
        for eps in EPS: add(f'scaled_e{eps:g}_d{d:02d}', base*eps*s/mx, set='scaled', eps=eps, d=d)
    th = theta_star()
    for sg, lab in ((1.0, 'plus'), (-1.0, 'minus')): add(f'gross_{lab}', sg*0.02*th, set='gross', sign=sg)
    add('true', np.zeros(9), set='true')
    return M


# --- variants -------------------------------------------------------------------------------------------------------------------
def variants(e0):
    V = {'V0': dict(kind='const', e=e0)}
    for D in MARGINS: V[f'VM({D:.2f})'] = dict(kind='const', e=e0+D, Delta=D)
    for T in FB_T:
        for m in FB_M: V[f'VF({T},{m})'] = dict(kind='fb', e0=e0, T=T, m=m)
    return V


# --- continuous-rule Monte Carlo job ------------------------------------------------------------------------------------------------
def run_job(args):
    """one (model, variant) with R trajectories and one seed; returns the per-step sums of c, the per-trajectory N and S, and (feedback)
    the sums of e_k and the counts at the clip bounds."""
    key, delta, var, R, seed = args
    xb = XK+HK@np.asarray(delta, float)
    rng = np.random.default_rng(seed)
    r = XK[0]+s*rng.standard_normal(R); N = np.ones(R); Wk = np.zeros(K); Wk[0] = float(R); S = np.ones(R)
    fb = var['kind'] == 'fb'
    if fb:
        qt = (B-var['m']-1.0)/(K-1.0); e0 = var['e0']; T = var['T']; esum = 0.0; esq = 0.0; nlo = 0; nhi = 0
    else:
        e = var['e']
    for p in range(1, K):
        if fb:
            k = p+1
            u = e0+(N-(1.0+qt*(k-1)))/(2.0*qt*T)
            nlo += int((u <= E_LO).sum()); nhi += int((u >= E_HI).sum())
            e = np.clip(u, E_LO, E_HI); esum += float(e.sum()); esq += float((e*e).sum())
        lo = np.minimum(xb[p]-e*s, r-s); hi = np.maximum(xb[p]+e*s, r+s)
        cev, cno = dr.cfun((lo-XK[p])/s, (hi-XK[p])/s); c = cev+cno
        Wk[p] = c.sum(); S += c
        zk = XK[p]+s*rng.standard_normal(R)
        snd = (zk <= lo) | (zk >= hi)
        N += snd; r = np.where(snd, zk, r)
    out = dict(key=key, Wk=Wk, N=N, S=S, R=R, seed=seed)
    if fb: out.update(esum=esum, esq=esq, nlo=nlo, nhi=nhi, ncount=R*(K-1))
    return out

def aggregate(outs):
    """record of one job (or of several batches of one job): E[N] (SE), sd of N, fraction N > B, S, log det J, Delta I; checks of §5 item 5."""
    N = np.concatenate([o['N'] for o in outs]); S = np.concatenate([o['S'] for o in outs]); R = N.size
    Wk = sum(o['Wk'] for o in outs)/R
    J = (HK*Wk[:, None]).T@HK/s**2; sgn, ld = np.linalg.slogdet(J)
    rec = dict(R=int(R), seeds=[o['seed'] for o in outs], EN=float(N.mean()), EN_se=float(N.std(ddof=1)/np.sqrt(R)), sd_N=float(N.std(ddof=1)),
               frac_over=float((N > B).mean()), S=float(S.mean()), S_se=float(S.std(ddof=1)/np.sqrt(R)), logdet=float(ld), dI=float(ld-I_SOD),
               W_min=float(Wk.min()), W_max=float(Wk.max()), N_min=float(N.min()), N_max=float(N.max()))
    if len(outs) > 1:
        ldb = [np.linalg.slogdet((HK*(o['Wk']/o['R'])[:, None]).T@HK/s**2)[1] for o in outs]
        rec['logdet_batch_se'] = float(np.std(ldb, ddof=1)/np.sqrt(len(outs)))
    if 'esum' in outs[0]:
        n = sum(o['ncount'] for o in outs); es = sum(o['esum'] for o in outs); eq = sum(o['esq'] for o in outs)
        rec.update(e_mean=es/n, e_sd=float(np.sqrt(max(eq/n-(es/n)**2, 0.0))), frac_clip_lo=sum(o['nlo'] for o in outs)/n,
                   frac_clip_hi=sum(o['nhi'] for o in outs)/n)
    rec['check5'] = bool(np.isfinite(rec['dI']) and sgn > 0 and rec['W_min'] >= 0.0 and rec['W_max'] <= 1.0 and rec['N_min'] >= 1.0)
    return rec


# --- mismatched lattice rule (checks 1, 2) ----------------------------------------------------------------------------------------
class MRule:
    """Round 14's R(n, j) with FL, CL of the prediction xbar = XK + HK delta: L_i = max(24, i - FLd[p] + n_p), U_i = max(24, CLd[p] - i + n_p)."""
    def __init__(self, delta, n=37, j=13):
        xb = XK+HK@np.asarray(delta, float); u = (xb-Z0)/DZ
        self.FLd = np.floor(u).astype(np.int64); self.CLd = np.ceil(u).astype(np.int64)
        pp = np.arange(K); self.n_p = np.where((pp % 20) < j, n, n+1).astype(np.int64)
        self._i = np.arange(G, dtype=np.int64); self._L = np.empty(G, np.int64); self._U = np.empty(G, np.int64)
        self.X = e13.X_PAD
        self.fl_shift = (self.FLd-e13.FL)                         # prediction error in grid steps (diagnostic)

    def rows_full(self, p, out=None):
        L, U = (self._L, self._U) if out is None else out
        npp = self.n_p[p]
        np.maximum(self._i-self.FLd[p]+npp, MIN_HW, out=L)
        np.maximum(self.CLd[p]-self._i+npp, MIN_HW, out=U)
        return L, U

    def is_sod(self): return False

def evaluate_mrule(rule, f=1):
    t0 = time.perf_counter()
    if f == 1: rec, J, fw = e13.evaluate(rule)
    else: rec, J, fw = e13.evaluate_refined(rule, f)
    rec['wall_s'] = time.perf_counter()-t0
    return rec, J

def _mc_lattice_m(args):
    """Round 14's routine 1c (rule14._mc_lattice = eval11._mc_batch with the rule's half-widths) with FL, CL replaced by FLd, CLd."""
    FLd, CLd, n_p, R, seed = args
    z0 = Z0; dz = DZ
    rng = np.random.default_rng(seed)
    r = XK[0]+s*rng.standard_normal(R); N = np.ones(R); Wk = np.zeros(K); Wk[0] = float(R); S = np.ones(R)
    nclip = 0
    for p in range(1, K):
        i = np.ceil((r-z0)/dz-0.5).astype(np.int64)
        nclip += int(((i < 0) | (i > G-1)).sum()); i = np.clip(i, 0, G-1)
        Lm = np.maximum(i-FLd[p]+n_p[p], MIN_HW); Up = np.maximum(CLd[p]-i+n_p[p], MIN_HW)
        lo = r-Lm*dz; hi = r+Up*dz
        cev, cno = dr.cfun((lo-XK[p])/s, (hi-XK[p])/s); c = cev+cno
        Wk[p] = c.sum(); S += c
        zk = XK[p]+s*rng.standard_normal(R)
        snd = (zk <= lo) | (zk >= hi)
        N += snd; r = np.where(snd, zk, r)
    return Wk, N, S, nclip

def mc_lattice_m(rule, R=10000, batches=20, seed=20261109, workers=4):
    from multiprocessing import Pool
    jobs = [(rule.FLd, rule.CLd, rule.n_p, R//batches, seed+b) for b in range(batches)]
    t0 = time.perf_counter()
    with Pool(workers) as pool: outs = pool.map(_mc_lattice_m, jobs)
    Ns = np.concatenate([x[1] for x in outs]); Ss = np.concatenate([x[2] for x in outs]); Wk = sum(x[0] for x in outs)/R
    J = (HK*Wk[:, None]).T@HK/s**2; ld = float(np.linalg.slogdet(J)[1])
    ldb = np.array([np.linalg.slogdet((HK*(x[0]/(R//batches))[:, None]).T@HK/s**2)[1] for x in outs])
    return dict(R=R, batches=batches, seed=seed, EN_mc=float(Ns.mean()), EN_se=float(Ns.std(ddof=1)/np.sqrt(R)), S_mc=float(Ss.mean()),
                S_se=float(Ss.std(ddof=1)/np.sqrt(R)), logdet_mc=ld, logdet_batch_se=float(ldb.std(ddof=1)/np.sqrt(batches)),
                n_clipped=int(sum(x[3] for x in outs)), wall_s=time.perf_counter()-t0)
