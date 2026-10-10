"""§8.5 stage 2: shared setup, exec'd by stage2_*.py.  Run from code_8_3_stage2/stage2_pkg/:
    exec(open('stage2_common.py').read())
Provides (all reused literally from the existing scripts, not rewritten):
  - mc83.py setup block via mc85_scan.py up to evaluate(): x (uniform), K, T, t, w, G, Gj, a0, sod, cfun, D, s,
    Nper, ti, regressors(a, chi, tt), output(a, chi)
  - qual(r) from mc85_post.py (8.3.6(g); the uniform reference qualifies by definition)
New here:
  - sod_batch(zz, D): numba version of mc83.sod for a block of trajectories (same comparisons, same order);
    check_trigger() proves it reproduces sod() exactly (eps, r, overshoots) on 100 trajectories; every stage
    script runs the check before using it.
  - evaluate_crn(a, chi, Zb): mc85_scan.evaluate on a given block of common draws Zb (same arithmetic, same
    summation order), plus per-batch log det, gap histogram, event rate q_k and w_k for stage-2 reporting.
  - summarize_s2(): per-candidate record (B, paired D +- se, decision, Delta m_D^(C),(B), paired batch se,
    nine generalized eigenvalues vs (C), m_E^(C) with paired batch se, m_E^(B), gaps, window AC powers).
    Batch se of m_E: sd over the nbatch per-batch values lambda_min(J_b, J_C,b) / sqrt(nbatch); the per-batch values
    are biased low relative to the full-R estimate (smaller samples), only their spread is used.
Parametrisation (protocol): p = softmax(z1,z2,z3,0), a_m = sqrt(p_m); chi continuous, reduced mod 2*pi only
when forming the input (here, inside evaluate_crn) and when saving."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np, json, time, sys, multiprocessing as mp
from scipy.linalg import eigh
from numba import njit

# --- setup block (mc83.py) + D, s, Nper, ti, regressors(), output() exactly as in mc85_scan.py ---
exec(open('mc85_scan.py').read().split('def evaluate(a, chi):')[0])
del R, rng, Z            # the stage-1 R=200 block defined there; stage-2 scripts build their own draw blocks
_post = open('mc85_post.py').read()
exec('def qual(r):' + _post.split('def qual(r):')[1].split('def counts')[0])   # mc85_post.py::qual, verbatim

OUT = 'stage2'; os.makedirs(OUT, exist_ok=True)
TWO_PI = 2*np.pi
a_u = np.full(4, 0.5); chi_u = np.zeros(4)
Hp = regressors(a_u, chi_u, ti); Jb = Hp.T@Hp/s**2; ldB = np.linalg.slogdet(Jb)[1]   # (B): periodic, N=316, uniform
PNAMES = ['G0', 'Re1', 'Im1', 'Re2', 'Im2', 'Re3', 'Im3', 'Re4', 'Im4']
theta0 = np.concatenate([[G(0)*a0], np.ravel(np.column_stack([Gj.real, Gj.imag]))])


class Tee:
    """copy stdout to a log file in stage2/ (every number in the report is traceable to these logs)."""
    def __init__(self, path): self.f = open(path, 'w'); self.o = sys.stdout
    def write(self, m): self.o.write(m); self.f.write(m); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()


def log_runtime(stage, t_start, note=''):
    el = time.time()-t_start
    line = f"{stage}\tstarted {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start))}\telapsed {el:.1f} s ({el/60:.2f} min)\t{note}\n"
    with open(f'{OUT}/runtime.log', 'a') as f: f.write(line)
    print('RUNTIME', line.strip())


# --- accelerated trigger -------------------------------------------------------------------------------
@njit(cache=False)
def _sod_batch(zz, Dl, eps, r):
    n, KK = zz.shape
    for i in range(n):
        last = zz[i, 0]; eps[i, 0] = True; r[i, 0] = np.nan
        for k in range(1, KK):
            r[i, k] = last
            if abs(zz[i, k]-last) >= Dl:
                eps[i, k] = True; last = zz[i, k]

def sod_batch(zz, Dl):
    """rows of zz are trajectories; returns event flags eps (n,K) and pre-update reference r (n,K), r[:,0]=nan,
    i.e. row-wise the first two outputs of mc83.sod. Overshoots are |z_k - r_k| - Delta at the events k>=1."""
    zz = np.ascontiguousarray(zz, dtype=np.float64)
    eps = np.zeros(zz.shape, np.bool_); r = np.empty(zz.shape); _sod_batch(zz, Dl, eps, r)
    return eps, r

def check_trigger(verbose=True):
    """Proof that sod_batch reproduces mc83.sod exactly on 100 trajectories: 4 inputs (uniform, the stage-1
    phase-only best, a tone-1-heavy and a tone-4-heavy LHS point) x 25 draws, rows 0..99 of seed 20261012."""
    Zc = np.random.default_rng(20261012).standard_normal((100, K))
    inputs = [(a_u, chi_u), (a_u, np.array([5, 0, 1, 4])*np.pi/3),
              (np.sqrt([0.467, 0.23, 0.177, 0.126]), np.array([0.3, 2.0, 4.1, 5.5])),
              (np.sqrt([0.1, 0.15, 0.25, 0.5]), np.array([1.0, 5.0, 2.5, 0.7]))]
    n_ok = 0; n_ev = 0
    for j, (a, chi) in enumerate(inputs):
        xc = output(a, chi); zz = xc+s*Zc[25*j:25*(j+1)]
        eb, rb = sod_batch(zz, D)
        for i in range(25):
            e0, r0, ov0 = sod(zz[i], D)
            ev = np.flatnonzero(eb[i])[1:]; ovb = np.abs(zz[i, ev]-rb[i, ev])-D
            same = np.array_equal(e0, eb[i]) and np.array_equal(r0, rb[i], equal_nan=True) and np.array_equal(ov0, ovb)
            n_ok += same; n_ev += int(e0.sum())
    msg = f"trigger check: sod_batch == mc83.sod (eps, r, overshoots, bitwise) on {n_ok}/100 trajectories ({n_ev} events)"
    if verbose: print(msg)
    assert n_ok == 100, msg
    return msg


# --- evaluation on a block of common draws ------------------------------------------------------------
def p_of_z(z):
    e = np.exp(np.concatenate([z, [0.0]])); return e/e.sum()

def evaluate_crn(a, chi, Zb, nbatch=10, chunk=200, extras=False):
    """mc85_scan.evaluate(a, chi) on the draw block Zb (same arithmetic and summation order), plus
    per-batch log det (nbatch equal batches, for paired batch se), pooled gap histogram (clock steps),
    and with extras=True the event rate q_k, x_k and w_k."""
    a = np.asarray(a, float); chi = np.mod(np.asarray(chi, float), TWO_PI)
    Hc = regressors(a, chi, t); xc = output(a, chi)
    Rn = Zb.shape[0]; bs = Rn//nbatch
    cnt = np.zeros(Rn); gapm = np.zeros(Rn); Wsum = np.zeros(K); Wb = np.zeros((nbatch, K))
    qsum = np.zeros(K); ghist = np.zeros(K, np.int64)
    for c0 in range(0, Rn, chunk):
        zz = xc+s*Zb[c0:c0+chunk]; eps, r = sod_batch(zz, D)
        al = (r[:, 1:]-D-xc[1:])/s; be = (r[:, 1:]+D-xc[1:])/s; cev, cno = cfun(al, be); cc = cev+cno
        for j in range(zz.shape[0]):
            i = c0+j; ev = np.where(eps[j])[0]; g = np.diff(ev)
            cnt[i] = eps[j].sum(); gapm[i] = g.mean(); ghist += np.bincount(g, minlength=K)
            c = np.concatenate([[1.0], cc[j]]); Wsum += c; Wb[i//bs] += c
        qsum += eps.sum(0)
    wk = Wsum/Rn; J = (Hc*wk[:, None]).T@Hc/s**2
    Jbat = np.array([(Hc*(Wb[b]/bs)[:, None]).T@Hc/s**2 for b in range(nbatch)])
    ldb = np.array([np.linalg.slogdet(Jbat[b])[1] for b in range(nbatch)])
    uac = xc-G(0)*a0; Eout = np.mean(uac**2)
    uin = sum(a[m]*np.cos(w[m]*t+chi[m]) for m in range(4)); Ein = np.mean(uin**2)
    res = dict(cnt=cnt, J=J, ldb=ldb, Jbat=Jbat, gap=float(np.mean(gapm)), ghist=ghist, Ein=float(Ein), Eout=float(Eout))
    if extras: res.update(q=qsum/Rn, x=xc, wk=wk)
    return res

def hist_quantiles(h, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    """inverted-CDF quantiles of the pooled gap distribution (clock steps) from its histogram."""
    c = np.cumsum(h); n = c[-1]; nz = np.flatnonzero(h)
    return dict(min=int(nz[0]), max=int(nz[-1]), mean=float((np.arange(h.size)*h).sum()/n),
                **{f"q{int(round(100*q)):02d}": int(np.searchsorted(c, q*n, side='left')) for q in qs})

def summarize_s2(cid, a, chi, res, ref, R_, seed_, src=''):
    """per-candidate record against the reference (C) = uniform input evaluated on the same draws."""
    a = np.asarray(a, float); chi_s = np.mod(np.asarray(chi, float), TWO_PI)
    Dr = res['cnt']-ref['cnt']; ld = np.linalg.slogdet(res['J'])[1]; ldC = np.linalg.slogdet(ref['J'])[1]
    lamC = eigh(res['J'], ref['J'], eigvals_only=True); lamB = eigh(res['J'], Jb, eigvals_only=True)
    dlb = res['ldb']-ref['ldb']; nb = dlb.size
    mEb = np.array([eigh(res['Jbat'][b], ref['Jbat'][b], eigvals_only=True)[0] for b in range(nb)])   # paired per-batch m_E
    r = dict(id=cid, src=src, R=R_, seed=seed_, a=a.tolist(), p=(a**2).tolist(), chi=chi_s.tolist(),
             B=float(res['cnt'].mean()), D=float(Dr.mean()), D_se=float(Dr.std(ddof=1)/np.sqrt(R_)),
             dmD_C=float(ld-ldC), dmD_B=float(ld-ldB), dmD_batch=float(dlb.mean()), dmD_se=float(dlb.std(ddof=1)/np.sqrt(nb)),
             mE_C=float(lamC[0]), mE_se=float(mEb.std(ddof=1)/np.sqrt(nb)), mE_batch=float(mEb.mean()), mE_B=float(lamB[0]), lamC=lamC.tolist(),
             gap=res['gap'], gapq=hist_quantiles(res['ghist']), Ein=res['Ein'], Eout=res['Eout'], ldb=res['ldb'].tolist())
    r['dec'] = qual(r)
    return r


# --- Pareto front (both coordinates maximised) ---------------------------------------------------------
def pareto_front(recs, kx='dmD_C', ky='mE_C'):
    """indices of the non-dominated records (maximise both); sweep in decreasing kx, ties by decreasing ky."""
    o = sorted(range(len(recs)), key=lambda i: (-recs[i][kx], -recs[i][ky])); f = []; best = -np.inf
    for i in o:
        if recs[i][ky] > best: f.append(i); best = recs[i][ky]
    return f

def is_uniform(r): return np.allclose(r['chi'], 0) and np.allclose(np.array(r['a'])**2, 0.25)

def fmt_p(r): return '(' + ','.join(f"{v:.3f}" for v in np.array(r['a'])**2) + ')'
def fmt_chi(r): return '(' + ','.join(f"{v:.3f}" for v in np.mod(r['chi'], TWO_PI)) + ')'

def pool(n=4): return mp.get_context('fork').Pool(n)


# --- parallel evaluation of many candidates on one block of common draws (fork: workers share Zb) --------
def _eval_worker(args):
    a, chi, extras = args
    return evaluate_crn(a, chi, _ZW, extras=extras)

def eval_many(cands, Zb, extras=False, nproc=4):
    global _ZW
    _ZW = Zb
    with pool(nproc) as pl: return pl.map(_eval_worker, [(c['a'], c['chi'], extras) for c in cands], chunksize=1)

def diff_se(r1, r2):
    """paired batch se of Delta m_D(r1) - Delta m_D(r2) (same draws, same batches)."""
    d = np.array(r1['ldb'])-np.array(r2['ldb']); return float(d.std(ddof=1)/np.sqrt(d.size))

def lam_table_row(r):
    return ' '.join(f"{v:7.3f}" for v in r['lamC'])
