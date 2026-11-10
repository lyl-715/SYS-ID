"""Round 16: random inputs (prompt reports_CN/prompt_round16_random_inputs_EN.md). Monte Carlo only: no DP, no grid evaluator.

Inputs (prompt §1). An input is (p, chi): powers p_m = a_m^2 on the four frequencies dr.w with sum p_m = 1, phases chi_m; a0 = 1, noise s,
K = 5000 unchanged. x = dr.output(a, chi), H = dr.regressors(a, chi, dr.t), a = sqrt(p).
  random set   rng = np.random.default_rng(20261113); repeat p = rng.dirichlet(ones(4)), chi = rng.uniform(0, 2 pi, 4); accept iff
               min(p) >= 0.02, until 200 are accepted (DECISION 16-1: the bound is the prompt's).
  anchors      U, SCH, P5514, LO, HI, N exactly as Round 13 registers them (round13/eval13.py at import: inputs13.json and its
               sources); x, H from dr.input_arrays(name) (chi reduced mod 2 pi), which is what eval13's design grid uses.
  derived      (i) phase-only: the first 50 accepted phase vectors with p = (1/4, 1/4, 1/4, 1/4); (ii) power-only: the first 50 accepted
               power vectors with zero phases.

Monte Carlo (prompt §2): the vectorised loop of round11/eval11._mc_batch (and round14/rule14._mc_cont): R trajectories, initial
r = x_1 + s * noise, first step always transmitted (N = 1, W_1 = 1), per step c = cev + cno from dr.cfun((lo - x_k)/s, (hi - x_k)/s),
noise z_k = x_k + s * noise drawn step by step, transmit iff z_k <= lo or z_k >= hi; one generator default_rng(seed) per run, the
same seed for every run and every input (common random numbers). W_k = mean over trajectories of c, J = s^-2 sum_k W_k h_k h_k^T.
  rule   lo = min(x_k - e s, r - s), hi = max(x_k + e s, r + s)          (the model is the true one: prediction = x_k)
  sod    lo = r - Delta, hi = r + Delta

Fits (prompt §2a, §2b): runs on a fixed list, extension until B is bracketed, linear interpolation to E[N] = B between the two adjacent
runs that bracket B (parameter linear in E[N]; log det J and S linear against E[N] between the same two runs).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
os.environ.setdefault('R13_INPUT', 'U')
import sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); R13 = os.path.join(EV, 'round13')
sys.path.insert(0, R13)
import numpy as np
import eval13 as e13          # imported for its module bindings only: ew, dr (round4/evaluator_window -> density_recursion_v1, with
                              # its working-directory handling) and the in-memory registration of the Round 13 inputs; nothing of
                              # eval13's grid evaluator is called
ew, dr = e13.ew, e13.dr
s, K, B = e13.s, e13.K, e13.B
assert B == 316.8921168493721 and K == 5000 and dr.a0 == 1.0
TWO_PI = 2*np.pi

SEED_INPUTS = 20261113
N_RANDOM = 200
P_MIN = 0.02                                   # DECISION 16-1 (prompt §1)
N_DERIVED = 50
ANCHORS = ('U', 'SCH', 'P5514', 'LO', 'HI', 'N')
SEED_NOISE = 20261113                           # common random numbers: every run, every input
SEEDS_FLOOR = tuple(range(20261114, 20261124))  # check 2: ten other seeds
R_MC = 2000
N_GROUPS = 10                                   # DECISION 16-4: W_k also accumulated in 10 contiguous groups of 200 (diagnostic only)
E_LIST0 = (1.52, 1.56, 1.60, 1.64); E_STEP = 0.04
D_LIST0 = (12.0, 16.0, 20.0, 24.0, 28.0, 34.0); D_STEP = 4.0
MAX_EXT = 30


class Stop(RuntimeError):
    """a check of round 16 failed: the run stops (no fallback)."""


def jdump(obj, path):
    tmp = path+'.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1, default=_jdef)
    os.replace(tmp, path)

def _jdef(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


# --- inputs ---------------------------------------------------------------------------------------------------------------
def draw_random_set(n=N_RANDOM, seed=SEED_INPUTS, pmin=P_MIN):
    """prompt §1: draws in the order p, chi per attempt; returns accepted (p, chi) in draw order and the number of rejected draws."""
    rng = np.random.default_rng(seed); acc = []; rej = 0; draws = 0
    while len(acc) < n:
        p = rng.dirichlet(np.ones(4)); chi = rng.uniform(0.0, TWO_PI, 4); draws += 1
        if p.min() >= pmin: acc.append((p, chi))
        else: rej += 1
    return acc, rej, draws

def make_input(name, p, chi, kind):
    """x, H from dr.output / dr.regressors with a = sqrt(p); the record of prompt §1."""
    p = np.asarray(p, np.float64); chi = np.asarray(chi, np.float64); a = np.sqrt(p)
    assert abs(float(np.sum(a**2))-1.0) <= 1e-12, (name, p)
    xk = dr.output(a, chi); H = dr.regressors(a, chi, dr.t)
    return dict(name=name, kind=kind, p=p, chi=chi, a=a, xk=xk, H=H)

def make_anchor(name):
    """anchor exactly as Round 13: a, chi from dr.CFG (registered by eval13), x, H from dr.input_arrays (chi mod 2 pi)."""
    e = dr.CFG['inputs'][name]; a = np.asarray(e['a'], np.float64); chi = np.mod(np.asarray(e['chi'], np.float64), TWO_PI)
    xk, H = dr.input_arrays(name)
    assert np.array_equal(xk, dr.output(a, chi)) and np.array_equal(H, dr.regressors(a, chi, dr.t)), name
    return dict(name=name, kind='anchor', p=a**2, chi=chi, a=a, xk=xk, H=H)

def describe(inp):
    xk, H = inp['xk'], inp['H']; dx = np.diff(xk)
    sgn, ld = np.linalg.slogdet(H.T@H/s**2); assert sgn > 0, inp['name']
    return dict(name=inp['name'], kind=inp['kind'], p=[float(v) for v in inp['p']], chi=[float(v) for v in inp['chi']],
                a=[float(v) for v in inp['a']], sum_p_minus_1=float(np.sum(inp['p'])-1.0),
                range_sigma=float((xk.max()-xk.min())/s), tv_sigma=float(np.abs(dx).sum()/s), max_step_sigma=float(np.abs(dx).max()/s),
                I_full=float(ld), max_p=float(np.max(inp['p'])), min_p=float(np.min(inp['p'])))


# --- Monte Carlo --------------------------------------------------------------------------------------------------------------
def mc_run(xk, H, kind, par, R=R_MC, seed=SEED_NOISE, ng=N_GROUPS):
    """one Monte Carlo run (prompt §2): kind 'rule' (par = e) or 'sod' (par = Delta/sigma). Random-number order of eval11._mc_batch."""
    t0 = time.perf_counter()
    assert R % ng == 0
    rng = np.random.default_rng(seed)
    r = xk[0]+s*rng.standard_normal(R); N = np.ones(R); S = np.ones(R)
    Wg = np.zeros((ng, K)); Wg[:, 0] = R//ng                        # W_1 = 1
    cmin, cmax = np.inf, -np.inf
    if kind == 'rule': e = float(par)
    elif kind == 'sod': Dl = float(par)*s
    else: raise ValueError(kind)
    for p in range(1, K):
        if kind == 'rule':
            lo = np.minimum(xk[p]-e*s, r-s); hi = np.maximum(xk[p]+e*s, r+s)
        else:
            lo = r-Dl; hi = r+Dl
        cev, cno = dr.cfun((lo-xk[p])/s, (hi-xk[p])/s); c = cev+cno
        Wg[:, p] = c.reshape(ng, -1).sum(1); S += c
        cmin = min(cmin, float(c.min())); cmax = max(cmax, float(c.max()))
        zk = xk[p]+s*rng.standard_normal(R)
        snd = (zk <= lo) | (zk >= hi)
        N += snd; r = np.where(snd, zk, r)
    W = Wg.sum(0)/R
    J = (H*W[:, None]).T@H/s**2; sgn, ld = np.linalg.slogdet(J)
    Ng = N.reshape(ng, -1).mean(1); ldg = np.array([np.linalg.slogdet((H*(Wg[g]/(R//ng))[:, None]).T@H/s**2)[1] for g in range(ng)])
    out = dict(kind=kind, par=float(par), seed=int(seed), R=int(R), EN=float(N.mean()), EN_se=float(N.std(ddof=1)/np.sqrt(R)),
               S=float(S.mean()), S_se=float(S.std(ddof=1)/np.sqrt(R)), logdet=float(ld), logdet_sign=float(sgn),
               logdet_group_se=float(ldg.std(ddof=1)/np.sqrt(ng)), EN_groups=Ng, logdet_groups=ldg,
               W_min=float(W.min()), W_max=float(W.max()), c_min=cmin, c_max=cmax, N_min=float(N.min()), wall_s=time.perf_counter()-t0)
    check_run(out)
    return out

def check_run(run):
    """check 4 (prompt §3): 0 <= W_k <= 1, N >= 1, log det J finite (and J positive definite)."""
    bad = []
    if not (run['W_min'] >= 0.0 and run['W_max'] <= 1.0): bad.append(f"W_k outside [0, 1]: [{run['W_min']!r}, {run['W_max']!r}]")
    if not run['N_min'] >= 1.0: bad.append(f"N_min {run['N_min']}")
    if not (np.isfinite(run['logdet']) and run['logdet_sign'] > 0): bad.append(f"log det J {run['logdet']} sign {run['logdet_sign']}")
    if bad: raise Stop(f"check 4, {run['kind']} par {run['par']!r} seed {run['seed']}: " + '; '.join(bad))


# --- fits ---------------------------------------------------------------------------------------------------------------------
def _bracket(runs, Bt):
    """the adjacent pair (in the parameter) whose E[N] bracket Bt; E[N] must be strictly decreasing in the parameter."""
    ps = sorted(runs); ens = np.array([runs[q]['EN'] for q in ps])
    if not np.all(np.diff(ens) < 0): raise Stop(f"E[N] not strictly decreasing in the parameter: {list(zip(ps, ens))}")
    idx = [i for i in range(len(ps)-1) if ens[i] >= Bt >= ens[i+1]]
    if len(idx) != 1: return None
    return ps[idx[0]], ps[idx[0]+1]

def _interp(runs, pa, pb, Bt):
    ra, rb = runs[pa], runs[pb]
    if not (ra['EN'] >= Bt >= rb['EN']): raise Stop(f"interpolation pair {pa}, {pb} does not bracket B: {ra['EN']}, {rb['EN']}")
    t = (Bt-ra['EN'])/(rb['EN']-ra['EN'])
    par = pa+t*(pb-pa); I = ra['logdet']+t*(rb['logdet']-ra['logdet']); S = ra['S']+t*(rb['S']-ra['S'])
    # diagnostic (DECISION 16-4): the same interpolation per group of 200 trajectories, same pair (may extrapolate slightly)
    ga, gb = np.asarray(ra['EN_groups']), np.asarray(rb['EN_groups'])
    tg = (Bt-ga)/(gb-ga); Ig = np.asarray(ra['logdet_groups'])+tg*(np.asarray(rb['logdet_groups'])-np.asarray(ra['logdet_groups']))
    pg = pa+tg*(pb-pa)
    return dict(pair=[pa, pb], EN_pair=[ra['EN'], rb['EN']], t=float(t), par_fit=float(par), I=float(I), S=float(S),
                I_group_se=float(Ig.std(ddof=1)/np.sqrt(len(Ig))), par_group_se=float(pg.std(ddof=1)/np.sqrt(len(pg))), brackets=True)

def fit_rule(inp, seed=SEED_NOISE, Bt=B, log=None):
    """prompt §2a: e in {1.52, 1.56, 1.60, 1.64}, extended by 0.04 on the side needed until B is bracketed; interpolation."""
    xk, H = inp['xk'], inp['H']; runs = {}
    def run(e):
        e = round(float(e), 6)
        if e not in runs: runs[e] = mc_run(xk, H, 'rule', e, seed=seed)
        return runs[e]
    for e in E_LIST0: run(e)
    ext = []
    while _bracket(runs, Bt) is None:
        ps = sorted(runs); ens = [runs[q]['EN'] for q in ps]
        if min(ens) > Bt: e = ps[-1]+E_STEP                           # every e sends too much: larger e
        elif max(ens) < Bt: e = ps[0]-E_STEP
        else: raise Stop(f"rule: B inside the E[N] range but no unique bracketing pair: {list(zip(ps, ens))}")
        if e <= 0 or len(ext) >= MAX_EXT: raise Stop(f"rule: cannot bracket B (e = {e})")
        ext.append(round(e, 6)); run(e)
    pa, pb = _bracket(runs, Bt); fit = _interp(runs, pa, pb, Bt)
    return dict(runs=[runs[q] for q in sorted(runs)], extended=ext, e_fit=fit['par_fit'], I_rule=fit['I'], S_rule=fit['S'], fit=fit)

def fit_sod(inp, seed=SEED_NOISE, Bt=B, log=None):
    """prompt §2b: Delta/sigma in {12, 16, 20, 24, 28, 34}, extended by 4 on the side needed until B is bracketed; then the two points at
    one third and two thirds of the bracketing interval; interpolation between the two closest bracketing values."""
    xk, H = inp['xk'], inp['H']; runs = {}
    def run(d):
        d = round(float(d), 9)
        if d not in runs: runs[d] = mc_run(xk, H, 'sod', d, seed=seed)
        return runs[d]
    for d in D_LIST0: run(d)
    ext = []
    while _bracket(runs, Bt) is None:
        ps = sorted(runs); ens = [runs[q]['EN'] for q in ps]
        if min(ens) > Bt: d = ps[-1]+D_STEP
        elif max(ens) < Bt: d = ps[0]-D_STEP
        else: raise Stop(f"sod: B inside the E[N] range but no unique bracketing pair: {list(zip(ps, ens))}")
        if d <= 0 or len(ext) >= MAX_EXT: raise Stop(f"sod: cannot bracket B (Delta/sigma = {d})")
        ext.append(round(d, 9)); run(d)
    da, db = _bracket(runs, Bt); w = db-da
    thirds = [da+w/3.0, da+2.0*w/3.0]
    for d in thirds: run(d)
    br = _bracket(runs, Bt)
    if br is None: raise Stop("sod: no bracketing pair after the one-third / two-thirds points")
    pa, pb = br; fit = _interp(runs, pa, pb, Bt)
    return dict(runs=[runs[q] for q in sorted(runs)], extended=ext, coarse_pair=[da, db], thirds=thirds, D_fit=fit['par_fit'],
                I_SoD=fit['I'], S_SoD=fit['S'], fit=fit)

def process(inp, seed=SEED_NOISE, Bt=B):
    """prompt §2 for one input: 2a, 2b, 2c."""
    t0 = time.perf_counter()
    d = describe(inp)
    fr = fit_rule(inp, seed, Bt); fs = fit_sod(inp, seed, Bt)
    I_full = d['I_full']
    der = dict(e_fit=fr['e_fit'], D_fit=fs['D_fit'], I_rule=fr['I_rule'], I_SoD=fs['I_SoD'], S_rule=fr['S_rule'], S_SoD=fs['S_SoD'],
               c_rule=fr['I_rule']-I_full, c_sod=fs['I_SoD']-I_full, gain=fr['I_rule']-fs['I_SoD'],
               info_per_send_rule=(fr['S_rule']-1.0)/(Bt-1.0), info_per_send_sod=(fs['S_SoD']-1.0)/(Bt-1.0),
               c_rule_group_se=fr['fit']['I_group_se'], c_sod_group_se=fs['fit']['I_group_se'],
               n_runs=len(fr['runs'])+len(fs['runs']))
    return dict(input=d, seed=int(seed), B=Bt, rule=fr, sod=fs, derived=der, wall_s=time.perf_counter()-t0)
