"""Round 18: accumulated information, the non-transmissions, and the estimator (prompt
reports_CN/prompt_round18_accumulated_information_EN.md). Input U, true model, B = 316.8921168493721. No DP, no grid evaluator.

Triggers (§1), all in continuous form, vectorised over trajectories with the random-number order of eval11._mc_batch
(r = x_1 + s noise for the forced first transmission, then one standard normal per trajectory and step):
  SOD      lo = r - Delta*, hi = r + Delta*
  RULE     lo = min(x_k - e s, r - s), hi = max(x_k + e s, r + s), e = 1.580 (Round 14, part1.json d.e_up)
  RULE_FB  RULE with Round 15's feedback VF(100, 3): e_k = clip(e + (N_k - (1 + q_t (k - 1)))/(2 q_t T), 0.5, 4.0),
           q_t = (B - m - 1)/(K - 1), T = 100, m = 3 (mismatch15.run_job with delta = 0)
  PRED     lo = x_k - c s, hi = x_k + c s, c = Phi^-1(1 - q/2), q = (B - 1)/(K - 1)
transmit iff z_k <= lo or z_k >= hi; c_k = sum of dr.cfun((lo - x_k)/s, (hi - x_k)/s).

Part A (§2): per trajectory J_cond = s^-2 (h_1 h_1^T + sum_{k>=2} c_k h_k h_k^T), J_sent = s^-2 sum_{k transmitted} h_k h_k^T.
Part B (§3): LS on the transmitted pairs (estimator_r8.ls_init) and the censored ML of rounds 8/8b (estimator_r8b.estimate_stage1:
  estimator_r8.run_lbfgsb from theta_LS, restart, round-8b acceptance) with the windows the receiver reconstructs from the record.
Part C (§4): the rule with e_k from the one-sided edge table E(rho): C0 constant, C1 terminal weights, C2 running information.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R13, R14, R15 = (os.path.join(EV, d) for d in ('round13', 'round14', 'round15'))
R8, R8B = os.path.join(EV, 'round8'), os.path.join(EV, 'round8b')
for _p in (R8B, R8, R15):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np
from scipy.stats import norm
import mismatch15 as m15                       # imports round13/eval13 (input U asserted) -> ew, dr
import estimator_r8 as e8
import estimator_r8b as e8b
import declared as dk

e13, dr = m15.e13, m15.dr
s, K, B = m15.s, m15.K, m15.B
XK, HK = m15.XK, m15.HK
D_STAR = e13.D
I_SOD = 197.20097668224076
assert m15.I_SOD == I_SOD and B == 316.8921168493721
assert dk.s == s and dk.D == D_STAR and np.array_equal(dk.H(), HK)
P = 9
E_RULE = float(json.load(open(os.path.join(R14, 'results', 'part1.json')))['d']['e_up'])
assert E_RULE == 1.58, E_RULE
FB_T, FB_M = 100, 3
E_LO, E_HI = 0.5, 4.0
Q_UNI = (B-1.0)/(K-1.0)
C_PRED = float(norm.isf(Q_UNI/2))               # Phi^-1(1 - q/2)
THETA_STAR = m15.theta_star()
HH = (HK[:, :, None]*HK[:, None, :]).reshape(K, P*P)   # h_k h_k^T flattened (unscaled)
TRIGGERS = ('SOD', 'RULE', 'RULE_FB', 'PRED')
UB_CERT = 207.71458782222572                    # round11b.json proposition5.UB (class with minimum distance 1 sigma, B + 0.25)
EPS_C2 = 0.02


def logdet(M):
    sg, ld = np.linalg.slogdet(M)
    assert np.all(sg > 0), 'log det of a non-positive-definite matrix'
    return ld

def J_of_weights(W):
    """s^-2 sum_k W_k h_k h_k^T (W per step, row 0 = h_1)."""
    return (HK*np.asarray(W, float)[:, None]).T@HK/s**2


# --- the trigger windows (§1) ---------------------------------------------------------------------------------------------
def window(trig, p, r, N):
    """(lo, hi, e) at array index p (step k = p + 1) for the trajectories' references r and counts N (transmissions so far)."""
    x = XK[p]
    if trig == 'SOD':
        return r-D_STAR, r+D_STAR, None
    if trig == 'PRED':
        lo = np.full(r.shape, x-C_PRED*s); hi = np.full(r.shape, x+C_PRED*s)
        return lo, hi, None
    if trig == 'RULE':
        e = E_RULE
    elif trig == 'RULE_FB':                     # mismatch15.run_job, 'fb' branch, delta = 0
        qt = (B-FB_M-1.0)/(K-1.0); k = p+1
        u = E_RULE+(N-(1.0+qt*(k-1)))/(2.0*qt*FB_T)
        e = np.clip(u, E_LO, E_HI)
    else:
        raise ValueError(trig)
    return np.minimum(x-e*s, r-s), np.maximum(x+e*s, r+s), e


# --- Part A: one batch of trajectories --------------------------------------------------------------------------------------
def batch_A(args):
    """trajectories of one trigger: per-step sums of c (Wk) and of the transmission indicator (Pk), per-trajectory N, S,
    log det J_cond, log det J_sent, and the sums of J_cond, J_sent over the batch."""
    trig, R, seed = args
    rng = np.random.default_rng(seed)
    r = XK[0]+s*rng.standard_normal(R); N = np.ones(R); S = np.ones(R)
    C = np.empty((K, R)); C[0] = 1.0
    Sn = np.empty((K, R)); Sn[0] = 1.0
    Wk = np.zeros(K); Wk[0] = float(R); Pk = np.zeros(K); Pk[0] = float(R)
    esum = esq = 0.0; nlo = nhi = 0
    for p in range(1, K):
        lo, hi, e = window(trig, p, r, N)
        if trig == 'RULE_FB':
            esum += float(e.sum()); esq += float((e*e).sum())
            nlo += int((e <= E_LO).sum()); nhi += int((e >= E_HI).sum())
        cev, cno = dr.cfun((lo-XK[p])/s, (hi-XK[p])/s); c = cev+cno
        C[p] = c; Wk[p] = c.sum(); S += c
        zk = XK[p]+s*rng.standard_normal(R)
        snd = (zk <= lo) | (zk >= hi)
        Sn[p] = snd; Pk[p] = float(snd.sum())
        N += snd; r = np.where(snd, zk, r)
    Jc = (C.T@HH).reshape(R, P, P)/s**2
    Js = (Sn.T@HH).reshape(R, P, P)/s**2
    out = dict(trig=trig, R=R, seed=seed, Wk=Wk, Pk=Pk, N=N, S=S, ld_cond=logdet(Jc), ld_sent=logdet(Js),
               Jc_sum=Jc.sum(0), Js_sum=Js.sum(0))
    if trig == 'RULE_FB': out.update(esum=esum, esq=esq, nlo=nlo, nhi=nhi, ncount=R*(K-1))
    return out


def summarise_A(outs):
    trig = outs[0]['trig']; R = sum(o['R'] for o in outs)
    N = np.concatenate([o['N'] for o in outs]); S = np.concatenate([o['S'] for o in outs])
    ldc = np.concatenate([o['ld_cond'] for o in outs]); lds = np.concatenate([o['ld_sent'] for o in outs])
    W = sum(o['Wk'] for o in outs)/R; Pw = sum(o['Pk'] for o in outs)/R
    J = J_of_weights(W); Jsm = J_of_weights(Pw)
    J_direct = sum(o['Jc_sum'] for o in outs)/R
    I = float(logdet(J)); ldS = float(logdet(Jsm))
    Ib = np.array([logdet(J_of_weights(o['Wk']/o['R'])) for o in outs])
    Isb = np.array([logdet(J_of_weights(o['Pk']/o['R'])) for o in outs])
    sd = float(ldc.std(ddof=1))
    corr = float(np.corrcoef(ldc, N)[0, 1]) if sd > 1e-9 and N.std() > 0 else None      # PRED: J_cond is the same for every run
    q = lambda x, a: float(np.percentile(x, a))
    rec = dict(trigger=trig, R=int(R), batches=len(outs), seeds=[o['seed'] for o in outs],
               I=I, dI=I-I_SOD, I_batch_se=float(Ib.std(ddof=1)/np.sqrt(len(outs))),
               I_direct=float(logdet(J_direct)), I_direct_minus_I=float(logdet(J_direct)-I),
               ld_run_mean=float(ldc.mean()), ld_run_sd=sd, ld_run_se=float(sd/np.sqrt(R)), ld_run_p05=q(ldc, 5), ld_run_p95=q(ldc, 95),
               ld_run_min=float(ldc.min()), ld_run_max=float(ldc.max()),
               I_minus_mean=float(I-ldc.mean()), corr_ld_N=corr,
               ld_sent_of_mean=ldS, ld_sent_of_mean_batch_se=float(Isb.std(ddof=1)/np.sqrt(len(outs))),
               value_nontx=float(I-ldS), value_nontx_batch_se=float((Ib-Isb).std(ddof=1)/np.sqrt(len(outs))),
               ld_sent_run_mean=float(lds.mean()), ld_sent_run_sd=float(lds.std(ddof=1)),
               S=float(S.mean()), S_se=float(S.std(ddof=1)/np.sqrt(R)), sum_W=float(W.sum()),
               EN=float(N.mean()), EN_se=float(N.std(ddof=1)/np.sqrt(R)), sd_N=float(N.std(ddof=1)), sum_P=float(Pw.sum()),
               frac_N_over_B=float((N > B).mean()), cond_J=float(np.linalg.cond(J)),
               W_min=float(W.min()), W_max=float(W.max()))
    if 'esum' in outs[0]:
        n = sum(o['ncount'] for o in outs); es = sum(o['esum'] for o in outs); eq = sum(o['esq'] for o in outs)
        rec.update(e_mean=es/n, e_sd=float(np.sqrt(max(eq/n-(es/n)**2, 0.0))), frac_clip_lo=sum(o['nlo'] for o in outs)/n,
                   frac_clip_hi=sum(o['nhi'] for o in outs)/n)
    return rec, J, W, Jsm


# --- Part B: experiments, records, estimators --------------------------------------------------------------------------------
def noise_B(i):
    """the K standard normals of experiment i (seed 20261116 + i): value 0 for z_1, value p for step p + 1."""
    return np.random.default_rng(20261116+i).standard_normal(K)

def simulate_B(trig, NZ):
    """all experiments of one trigger at once (rows of NZ = experiments). Returns the transmission pattern ev (bool), the
    measurements z (all steps; the record keeps only the transmitted ones) and the sensor's windows lo, hi (nan at step 1)."""
    R = NZ.shape[0]
    z = XK[None, :]+s*NZ
    ev = np.zeros((R, K), bool); ev[:, 0] = True
    lo_all = np.full((R, K), np.nan); hi_all = np.full((R, K), np.nan)
    r = z[:, 0].copy(); N = np.ones(R)
    for p in range(1, K):
        lo, hi, _ = window(trig, p, r, N)
        zk = z[:, p]
        snd = (zk <= lo) | (zk >= hi)
        ev[:, p] = snd; lo_all[:, p] = lo; hi_all[:, p] = hi
        N += snd; r = np.where(snd, zk, r)
    return ev, z, lo_all, hi_all

def receiver_windows(trig, ev, zt):
    """the windows (lo_k, hi_k) of the non-transmitted steps as the receiver reconstructs them: r_k from the record
    (estimator_r8.references: the last transmitted value), x_k = the design output (here the true XK)."""
    r = e8.references(ev, zt); kk = np.flatnonzero(~ev)
    rn = r[kk]; x = XK[kk]
    if trig == 'SOD': return rn-D_STAR, rn+D_STAR
    if trig == 'PRED': return x-C_PRED*s, x+C_PRED*s
    if trig == 'RULE': return np.minimum(x-E_RULE*s, rn-s), np.maximum(x+E_RULE*s, rn+s)
    raise ValueError(trig)

def stage_data(trig, ev, zt):
    """likelihood data with the fields estimator_r8.Objective / ls_init read (He, ze, Hn, lo, up, N)."""
    if trig == 'SOD':
        return e8.StageData(ev, zt, ('sod',))
    d = e8.StageData.__new__(e8.StageData)
    Hm = dk.H(); ne = ~ev
    d.ev = ev; d.N = int(ev.sum()); d.He = Hm[ev]; d.ze = zt[ev]; d.Hn = Hm[ne]
    d.lo, d.up = receiver_windows(trig, ev, zt)
    assert np.all(np.isfinite(d.ze)) and np.all(np.isfinite(d.lo)) and np.all(np.isfinite(d.up))
    return d

def estimate_B(args):
    """one experiment: LS and CML (estimator_r8b.estimate_stage1, restart seed 80261116 + i)."""
    trig, i, ev, zrec = args
    t0 = time.perf_counter()
    zt = np.where(ev, zrec, np.nan)                    # the record: transmitted values only
    d = stage_data(trig, ev, zt)
    if trig == 'SOD':                                  # check: the receiver's SoD windows are those of StageData
        lo2, up2 = receiver_windows('SOD', ev, zt); assert np.array_equal(lo2, d.lo) and np.array_equal(up2, d.up)
    res = e8b.estimate_stage1(d, 80261116+i)
    a1, a2 = res['attempts']
    return dict(trig=trig, i=i, N=int(ev.sum()), theta_LS=np.asarray(res['theta_LS'], float), LS_rank=res['LS_rank'],
                LS_usable=res['LS_usable'], theta_CML=np.asarray(res['theta'], float), converged=bool(res['converged']),
                used=res['used'], fallback=res['fallback'], acc1=bool(a1['accept']['supp']), acc2=bool(a2['accept']['supp']),
                nit1=int(a1['nit']), decr1=float(a1['decrement']), msg1=a1['message'], success1=bool(a1['success']),
                wall_s=time.perf_counter()-t0)

def acc_stats(th, J):
    """bias, RMSE in Cramer-Rao units, Mahalanobis norm of the mean error, D-hat, -log det of the sample covariance."""
    e = th-THETA_STAR[None, :]; n = e.shape[0]
    crsd = np.sqrt(np.diag(np.linalg.inv(J)))
    b = e.mean(0); sd = e.std(0, ddof=1)
    M = e.T@e/n; Cv = np.cov(th.T, ddof=1)
    bias = b/crsd; bias_se = sd/np.sqrt(n)/crsd; rmse = np.sqrt((e*e).mean(0))/crsd
    return dict(n=int(n), cr_sd=crsd.tolist(), bias_cr=bias.tolist(), bias_cr_se=bias_se.tolist(),
                bias_max_abs=float(np.abs(bias).max()), bias_argmax=int(np.abs(bias).argmax()),
                bias_mahalanobis=float(np.sqrt(b@J@b)), rmse_cr=rmse.tolist(), rmse_cr_min=float(rmse.min()), rmse_cr_max=float(rmse.max()),
                sd_cr=(sd/crsd).tolist(), D_hat=float(-logdet(M)), neg_logdet_cov=float(-logdet(Cv)))


# --- Part C: the one-sided edge table and the three variants ------------------------------------------------------------------
E_GRID = (100+np.arange(701))/200.0            # 0.5, 0.505, ..., 4.0
assert E_GRID[0] == 0.5 and E_GRID[-1] == 4.0 and np.allclose(np.diff(E_GRID), 0.005)

def q_one(e): return norm.cdf(-e)
def c_one(e): return norm.cdf(-e)+e*norm.pdf(e)+norm.pdf(e)**2/norm.cdf(e)

def edge_table():
    """upper envelope of the lines rho -> rho c_one(e_i) - q_one(e_i) over the grid: breakpoints bp (ascending) and the edge
    E on each interval: E(rho) = ev[searchsorted(bp, rho, 'right')] (rho < bp[0] -> ev[0], ...). Ties at a breakpoint go to
    the interval on the right (the smaller e)."""
    c = c_one(E_GRID); q = q_one(E_GRID)
    o = np.argsort(c, kind='stable')                 # ascending slope (c decreases with e)
    hull = []                                        # indices into the grid; convex-hull trick for the maximum
    def x_int(i, j): return (q[j]-q[i])/(c[j]-c[i])  # rho at which line j overtakes line i (c[j] > c[i])
    for j in o:
        if hull and c[hull[-1]] == c[j]:
            if -q[j] <= -q[hull[-1]]: continue
            hull.pop()
        while len(hull) >= 2 and x_int(hull[-2], j) <= x_int(hull[-2], hull[-1]): hull.pop()
        hull.append(j)
    hull = np.array(hull)
    bp = np.array([x_int(hull[t], hull[t+1]) for t in range(len(hull)-1)])
    assert np.all(np.diff(bp) > 0)
    return bp, E_GRID[hull], hull, c, q

BP, EV_HULL, HULL, C1E, Q1E = edge_table()

def E_of(rho):
    return EV_HULL[np.searchsorted(BP, rho, side='right')]

def E_brute(rho):
    """direct argmax over the grid (first maximum = smallest e), for the check."""
    v = np.asarray(rho, float)[:, None]*C1E[None, :]-Q1E[None, :]
    return E_GRID[np.argmax(v, axis=1)]


def gamma_terminal(Jref):
    """C1: gamma_k = s^-2 h_k^T J_ref^-1 h_k."""
    Ji = np.linalg.inv(Jref)
    return np.einsum('ki,ij,kj->k', HK, Ji, HK)/s**2

def run_C(args):
    """one run of a Part C variant with R trajectories and one noise seed (random-number order of eval11._mc_batch).
    variant 'C0': par = e; 'C1': par = lambda (e_k = E(gamma_k / lambda)); 'C2': par = lambda (running information)."""
    var, par, R, seed, Jref, Sref = args
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    r = XK[0]+s*rng.standard_normal(R); N = np.ones(R); S = np.ones(R)
    C = np.empty((K, R)); C[0] = 1.0
    Wk = np.zeros(K); Wk[0] = float(R)
    ecount = np.zeros(E_GRID.size, np.int64); esum = 0.0
    if var == 'C1':
        ek_all = E_of(gamma_terminal(Jref)/par)
    if var == 'C2':
        Mt = np.broadcast_to(EPS_C2*s**2*Jref, (R, P, P)).copy()+np.outer(HK[0], HK[0])[None]   # s^2 M_2
        St = np.full(R, EPS_C2*Sref+1.0)
        rel_sum = 0.0; rho_min = np.inf; rho_max = 0.0
    for p in range(1, K):
        x = XK[p]
        if var == 'C0':
            e = par
        elif var == 'C1':
            e = ek_all[p]
        else:
            h = HK[p]
            g = np.linalg.solve(Mt, np.broadcast_to(h[:, None], (R, P, 1)))[:, :, 0]@h      # h^T (s^2 M_k)^-1 h = s^-2 h^T M_k^-1 h
            rel = g*St/P; rel_sum += float(rel.sum())
            rho = rel*(P/Sref)/par; rho_min = min(rho_min, float(rho.min())); rho_max = max(rho_max, float(rho.max()))
            e = E_of(rho)
        if var == 'C1':
            ecount[int(round((e-0.5)*200))] += R; esum += e*R
        elif var == 'C2':
            ecount += np.bincount(np.rint((e-0.5)*200).astype(np.int64), minlength=E_GRID.size); esum += float(e.sum())
        lo = np.minimum(x-e*s, r-s); hi = np.maximum(x+e*s, r+s)
        cev, cno = dr.cfun((lo-x)/s, (hi-x)/s); c = cev+cno
        C[p] = c; Wk[p] = c.sum(); S += c
        if var == 'C2':
            Mt += c[:, None, None]*np.outer(HK[p], HK[p])[None]; St += c
        zk = x+s*rng.standard_normal(R)
        snd = (zk <= lo) | (zk >= hi)
        N += snd; r = np.where(snd, zk, r)
    Jc = (C.T@HH).reshape(R, P, P)/s**2
    ldc = logdet(Jc)
    W = Wk/R; J = J_of_weights(W); I = float(logdet(J))
    out = dict(var=var, par=float(par), R=R, seed=seed, EN=float(N.mean()), EN_se=float(N.std(ddof=1)/np.sqrt(R)), sd_N=float(N.std(ddof=1)),
               I=I, dI=I-I_SOD, ld_run_mean=float(ldc.mean()), ld_run_sd=float(ldc.std(ddof=1)), S=float(S.mean()),
               wall_s=0.0)
    if var == 'C0':
        out.update(e_p05=float(par), e_p50=float(par), e_p95=float(par), e_mean=float(par))
    else:
        cw = np.cumsum(ecount)/ecount.sum()
        qe = lambda a: float(E_GRID[np.searchsorted(cw, a)])                 # smallest e with cumulative share >= a
        out.update(e_p05=qe(0.05), e_p50=qe(0.50), e_p95=qe(0.95), e_mean=esum/ecount.sum(),
                   e_frac_lo=float(ecount[0]/ecount.sum()), e_frac_hi=float(ecount[-1]/ecount.sum()))
    if var == 'C2':
        Jc2 = (Mt-EPS_C2*s**2*Jref[None])/s**2                               # s^2 M after the last step, minus eps J_ref: = J_cond
        out.update(rel_mean=rel_sum/(R*(K-1)), rho_min=rho_min, rho_max=rho_max,
                   Jc_running_vs_direct=float(np.abs(Jc2-Jc).max()/np.abs(Jc).max()))
    out['wall_s'] = time.perf_counter()-t0
    return out
