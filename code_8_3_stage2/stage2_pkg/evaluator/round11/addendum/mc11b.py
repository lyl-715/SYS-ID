"""Round 11b: Monte Carlo at the runtime lookup rule for tables of the union action set (DECISION 11-7 of round 11, unchanged).

_mc_batch_u is eval11._mc_batch with one change: the half-widths of the looked-up node come from the union action set (per-step edge
nodes LN, UN, block minimum minv; sprime.ActionSetU) instead of eval11's am (i - fl) + bm with the fixed 48 clamp. Everything else is
eval11's: R0 = x_1 + sigma xi, nearest node (ties -> smaller index), window (r - L h, r + U h) centred at the continuous r, c from
dr.cfun, send iff z_k <= lo or z_k >= hi, r <- z_k on a send; same random-number order, seeds seed + batch. monte_carlo_u aggregates
exactly as eval11.monte_carlo (E[N], S per trajectory; J from the mean per-step weights; log det standard error from the 20 batch log
dets) and adds the z-scores against the sigma/120 values (pass rule |z| <= 3 and no clipped lookups, DECISION 11-7).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
import sprime as sp
e11, dr = sp.e11, sp.dr
s, K, G = sp.s, sp.K, sp.G
XK, HK, Z0, DZ = sp.XK, sp.HK, sp.Z0, sp.DZ

_ACTS = None
def _acts():
    global _ACTS
    if _ACTS is None: _ACTS = sp.set_union()
    return _ACTS

def _mc_batch_u(args):
    codes_path, R, seed = args
    a = _acts(); z0 = Z0; dz = DZ
    rng = np.random.default_rng(seed)
    codes = np.load(codes_path, mmap_mode='r')
    r = XK[0]+s*rng.standard_normal(R); N = np.ones(R); Wk = np.zeros(K); Wk[0] = float(R); S = np.ones(R)
    nclip = 0
    for p in range(1, K):
        i = np.ceil((r-z0)/dz-0.5).astype(np.int64)                     # nearest node, ties -> smaller index
        nclip += int(((i < 0) | (i > G-1)).sum()); i = np.clip(i, 0, G-1)
        cd = np.asarray(codes[p-1][i], np.int64)
        b = np.searchsorted(a.boff, cd, side='right')-1; nb = a.bhi[b]-a.blo[b]; rr = cd-a.boff[b]
        jl = a.blo[b]+rr//nb; jh = a.blo[b]+rr % nb
        Lm = np.maximum(a.am[jl]*(i-a.LN[p, jl])+a.bmin[jl], a.minv[jl]); Up = np.maximum(a.am[jh]*(a.UN[p, jh]-i)+a.bmin[jh], a.minv[jh])
        lo = r-Lm*dz; hi = r+Up*dz
        cev, cno = dr.cfun((lo-XK[p])/s, (hi-XK[p])/s); c = cev+cno
        Wk[p] = c.sum(); S += c
        zk = XK[p]+s*rng.standard_normal(R)
        snd = (zk <= lo) | (zk >= hi)
        N += snd; r = np.where(snd, zk, r)
    return Wk, N, S, nclip

def monte_carlo_u(codes_path, rec24, rec120, R=10000, batches=20, seed=20261109, workers=4, zmax=3.0):
    from multiprocessing import Pool
    _acts()
    jobs = [(codes_path, R//batches, seed+b) for b in range(batches)]
    with Pool(workers) as pool: outs = pool.map(_mc_batch_u, jobs)
    Ns = np.concatenate([x[1] for x in outs]); Ss = np.concatenate([x[2] for x in outs]); Wk = sum(x[0] for x in outs)/R
    J = (HK*Wk[:, None]).T@HK/s**2; ld = float(np.linalg.slogdet(J)[1])
    ldb = np.array([np.linalg.slogdet((HK*(x[0]/(R//batches))[:, None]).T@HK/s**2)[1] for x in outs])
    d = dict(R=R, batches=batches, seed=seed, EN_mc=float(Ns.mean()), EN_se=float(Ns.std(ddof=1)/np.sqrt(R)), S_mc=float(Ss.mean()),
             S_se=float(Ss.std(ddof=1)/np.sqrt(R)), logdet_mc=ld, logdet_batch_se=float(ldb.std(ddof=1)/np.sqrt(batches)),
             EN_eval=rec24['EN'], S_eval=rec24['S'], logdet_eval=rec24['logdet'], n_clipped=int(sum(x[3] for x in outs)))
    d['z_EN'] = (d['EN_eval']-d['EN_mc'])/d['EN_se']; d['z_S'] = (d['S_eval']-d['S_mc'])/d['S_se']
    d['z_logdet'] = (d['logdet_eval']-d['logdet_mc'])/d['logdet_batch_se']
    if rec120 is not None:
        d['EN_eval120'] = rec120['EN']; d['S_eval120'] = rec120['S']; d['logdet_eval120'] = rec120['logdet']
        d['z120_EN'] = (rec120['EN']-d['EN_mc'])/d['EN_se']; d['z120_S'] = (rec120['S']-d['S_mc'])/d['S_se']
        d['z120_logdet'] = (rec120['logdet']-d['logdet_mc'])/d['logdet_batch_se']
        d['pass_'] = bool(abs(d['z120_EN']) <= zmax and abs(d['z120_S']) <= zmax and abs(d['z120_logdet']) <= zmax and d['n_clipped'] == 0)
    d['pass_sigma24'] = bool(abs(d['z_EN']) <= zmax and abs(d['z_S']) <= zmax and abs(d['z_logdet']) <= zmax)
    d['criterion'] = '|z| <= 3 for E[N], S, log det against the sigma/120 values and no clipped lookups (DECISION 11-7); sigma/24 z-scores as pass_sigma24'
    return d
