"""Route 1 (free-weight D-optimal relaxation): shared setup, exec'd by bound/bound_*.py.
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bound_common.py')).read())
The working directory is switched to code_8_3_stage2/stage2_pkg/, so the stage scripts' relative paths hold.

Reused literally (exec of their source, not rewritten):
  - mc83.py setup block via mc85_scan.py up to its random draws: system G, tones w, clock grid t (K = 5000, dt = 1 ms),
    a0, x (uniform input), H (uniform-input regressors), D = Delta_*, s = sigma = Delta_*/20, Nper = 316, ti = periodic times,
    regressors(a, chi, tt), output(a, chi). The R = 200 draw block of mc85_scan.py is NOT executed (no Monte Carlo here).
  - p_of_z from stage2_common.py, x_to_input / input_to_x from stage2_local.py (stage-2 parametrisation
    p = softmax(z1, z2, z3, 0), a = sqrt(p), chi continuous).
New here (deterministic):
  - ld(Hs, xi): log det(sigma^-2 sum_k xi_k h_k h_k^T).
  - proj_capped(y, W, cap): Euclidean projection onto {0 <= xi <= cap, sum xi = W} (cap = inf: scaled simplex).
  - top_sum(d, W, cap): max of sum_k xi_k d_k over Xi_W = {0 <= xi <= cap, sum xi = W} (an LP over the box-and-sum set,
    solved by sorting: the floor(W/cap) largest d_k at cap plus the remainder on the next one).
  - certificate(Hs, xi, W, cap): for a feasible xi~ with F = log det(sigma^-2 M), d_k = h_k^T M^-1 h_k = dF/dxi_k:
      cutting-plane bound (QUOTED):  U_cp = F(xi~) + max_{xi in Xi_W} grad F(xi~)^T (xi - xi~) = F + top_sum(d) - xi~^T d,
        valid because F is concave (F(xi) <= F(xi~) + grad^T (xi - xi~) for every xi);
      duality bound (secondary, tighter): U_dual = F + p log(top_sum(d)/p), p = 9, from log det M' <= tr(Y M') - log det Y - p
        with Y = (p/top_sum) M^-1 (xi~^T d = p).  U_dual <= U_cp since log(1+x) <= x. Both gaps are zero exactly at the optimum
        (KKT: d_k >= nu on xi_k = cap, = nu on interior, <= nu on xi_k = 0).
  - spg(Hs, W, cap, xi0, tol): spectral projected gradient (Birgin-Martinez-Raydan, nonmonotone line search, BB steps),
    stopped when the cutting-plane gap <= tol (nat).
  - cvx_solve(Hs, W, cap, solver): the same problem in cvxpy with log_det.
"""
import os, sys, time, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PKG)
import numpy as np

# --- setup block (mc83.py via mc85_scan.py), stopping before mc85_scan's random draws --------------------------------
_scan = open('mc85_scan.py').read()
exec(_scan.split('rng = np.random.default_rng(20261010)')[0])                     # imports, mc83 setup, D, s, R (unused)
exec(_scan[_scan.index('Nper = 316'):_scan.index('def evaluate(a, chi):')])         # Nper, ti, regressors(), output()
del R
_c2 = open('stage2_common.py').read()
exec(_c2[_c2.index('def p_of_z(z):'):_c2.index('def evaluate_crn(')])               # p_of_z (stage 2)
TWO_PI = 2*np.pi
_l2 = open('stage2_local.py').read()
exec(_l2[_l2.index('def x_to_input(x):'):_l2.index('def eval_s(x):')])              # x_to_input, input_to_x (stage 2)

OUT = 'bound'; os.makedirs(OUT, exist_ok=True)
P = 9                                                   # number of parameters
LOG_S2 = 2*np.log(s)                                    # log sigma^2
a_u = np.full(4, 0.5); chi_u = np.zeros(4)
H_u = regressors(a_u, chi_u, t)                         # uniform-input regressors on the clock grid (= mc83 H)
WK_SOD = np.load('wk_d20_10000.npy')                    # send-on-delta weights of the uniform input (mc837.py, R = 10000)
LD_PER = float(np.linalg.slogdet(regressors(a_u, chi_u, ti).T@regressors(a_u, chi_u, ti)/s**2)[1])   # 194.155
LD_SOD = float(np.linalg.slogdet((H_u*WK_SOD[:, None]).T@H_u/s**2)[1])                               # 197.197
W_A, W_C = 316.0, 316.0
W_B = float(WK_SOD.sum())                               # review (2): B uses the input's own S(u); uniform input: 465.958
VARIANTS = {'A': (W_A, 1.0), 'B': (W_B, 1.0), 'C': (W_C, np.inf)}


class Tee:
    """copy stdout to a log file in bound/ (every number in the report is traceable to these logs)."""
    def __init__(self, path): self.f = open(path, 'w'); self.o = sys.stdout
    def write(self, m): self.o.write(m); self.f.write(m); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()


def log_runtime(stage, t_start, note=''):
    el = time.time()-t_start
    line = f"{stage}\tstarted {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start))}\telapsed {el:.1f} s ({el/60:.2f} min)\t{note}\n"
    with open(f'{OUT}/runtime.log', 'a') as f: f.write(line)
    print('RUNTIME', line.strip())


def axis_ratio(dld): return float(np.exp(-dld/18.0))     # geometric-mean axis ratio of the 9-dim confidence ellipsoids


# --- objective, gradient, certificate -----------------------------------------------------------------------------------
def ld(Hs, xi):
    return float(np.linalg.slogdet((Hs*xi[:, None]).T@Hs)[1] - P*LOG_S2)

def ld_grad(Hs, xi):
    """log det(sigma^-2 M) and d_k = h_k^T M^-1 h_k (= gradient w.r.t. xi_k)."""
    M = (Hs*xi[:, None]).T@Hs
    L = np.linalg.cholesky(M)
    V = np.linalg.solve(L, Hs.T)                         # 9 x K
    return float(2*np.log(np.diag(L)).sum() - P*LOG_S2), np.einsum('ij,ij->j', V, V)

def top_sum(d, W, cap):
    """max sum_k xi_k d_k over {0 <= xi <= cap, sum xi = W}."""
    if not np.isfinite(cap): return float(W*d.max())
    n = int(np.floor(W/cap)); ds = -np.sort(-d)
    return float(cap*ds[:n].sum() + (W - n*cap)*(ds[n] if n < d.size else 0.0))

def certificate(Hs, xi, W, cap, d=None, f=None):
    """(F, cutting-plane upper bound U_cp, gap_cp) for a feasible xi; certificate_full also returns the duality bound."""
    if d is None: f, d = ld_grad(Hs, xi)
    S = top_sum(d, W, cap); gap_cp = S - float(xi@d)
    return f, f+gap_cp, gap_cp

def certificate_full(Hs, xi, W, cap):
    f, d = ld_grad(Hs, xi); S = top_sum(d, W, cap)
    return dict(F=f, U_cp=f+S-float(xi@d), gap_cp=S-float(xi@d), U_dual=f+P*np.log(S/P), gap_dual=P*np.log(S/P), xid=float(xi@d))

def proj_capped(y, W, cap):
    """Euclidean projection of y onto {0 <= xi <= cap, sum xi = W}: xi = clip(y - tau, 0, cap)."""
    hi_c = cap if np.isfinite(cap) else np.inf
    lo, hi = y.min() - (cap if np.isfinite(cap) else W) - 1.0, y.max()
    for _ in range(100):                                  # bisection on tau (sum is non-increasing in tau)
        tau = 0.5*(lo+hi)
        if np.clip(y-tau, 0.0, hi_c).sum() > W: lo = tau
        else: hi = tau
    tau = 0.5*(lo+hi); z = y-tau
    free = (z > 0) & (z < hi_c)                           # exact finish on the free set
    if free.any():
        n_up = int((z >= hi_c).sum()) if np.isfinite(cap) else 0
        tau = (y[free].sum() + n_up*(cap if np.isfinite(cap) else 0.0) - W)/free.sum()
    return np.clip(y-tau, 0.0, hi_c)

def kkt_summary(xi, d, cap, tol=1e-9):
    """interior count, KKT level nu and the largest sign violation of the KKT conditions (in units of d)."""
    zero = xi <= tol; full = (xi >= cap-tol) if np.isfinite(cap) else np.zeros(xi.size, bool); inter = ~zero & ~full
    if inter.any(): nu = float(np.median(d[inter]))
    else: nu = float(0.5*(d[zero].max() + (d[full].min() if full.any() else d[zero].max())))
    viol = max(float((d[zero]-nu).max()) if zero.any() else -np.inf,
               float((nu-d[full]).max()) if full.any() else -np.inf,
               float(np.abs(d[inter]-nu).max()) if inter.any() else -np.inf)
    return dict(n_interior=int(inter.sum()), n_full=int(full.sum()), n_support=int((~zero).sum()), nu=nu,
                kkt_viol=max(viol, 0.0), kkt_viol_rel=max(viol, 0.0)/nu)


# --- own first-order method: spectral projected gradient ----------------------------------------------------------------
def spg(Hs, W, cap, xi0=None, tol=1e-8, maxit=20000, mem=10, verbose=False):
    """maximise log det(sigma^-2 sum xi_k h_k h_k^T) over {0 <= xi <= cap, sum xi = W}. Stops when the cutting-plane gap <= tol."""
    K_ = Hs.shape[0]
    xi = proj_capped(np.full(K_, W/K_) if xi0 is None else np.asarray(xi0, float), W, cap)
    f, g = ld_grad(Hs, xi); hist = [f]
    alpha = 1.0/np.abs(g).max(); it = 0; nfg = 1
    while True:
        S = top_sum(g, W, cap); gap = S - float(xi@g)          # cutting-plane gap
        if gap <= tol or it >= maxit: break
        dvec = proj_capped(xi + alpha*g, W, cap) - xi
        gd = float(g@dvec); fref = max(hist[-mem:]); lam = 1.0
        while True:                                       # nonmonotone Armijo (maximisation)
            xn = xi + lam*dvec
            try:
                fn, gn = ld_grad(Hs, xn); nfg += 1
            except np.linalg.LinAlgError:
                fn = -np.inf
            if fn >= fref + 1e-4*lam*gd or lam < 1e-12: break
            lam *= 0.5
        s_ = xn-xi; y_ = gn-g; sy = -float(s_@y_)      # concave: -s'y >= 0
        alpha = float(s_@s_)/sy if sy > 0 else 1e3/np.abs(gn).max()
        alpha = min(max(alpha, 1e-12), 1e12)
        xi, f, g = xn, fn, gn; hist.append(f); it += 1
        if verbose and it % 200 == 0: print(f'   spg it {it}: f={f:.10f} gap={gap:.2e}')
    return dict(xi=xi, f=f, d=g, ub=f+gap, gap=gap, it=it, nfg=nfg)


# --- own first-order method 2: batched pairwise exchange (vertex-exchange / REX type) ------------------------------------
# Uses only the gradient d_k = h_k^T M^-1 h_k and an exact line search along xi + delta (e_i - e_j):
#   det(M + delta(h_i h_i^T - h_j h_j^T)) / det M = (1 + delta d_i)(1 - delta d_j) + delta^2 d_ij^2,
#   maximised at delta* = (d_i - d_j) / (2 (d_i d_j - d_ij^2)), clipped to [0, min(cap - xi_i, xi_j)].
# Each sweep: recompute M^-1 and d from xi (no drift), stop if the cutting-plane gap <= tol, else pair the `batch` largest-d
# non-full points with the `batch` smallest-d support points (the leading pair first, which alone gives the classical
# vertex-exchange guarantee) and apply the exchanges sequentially with fresh d_i, d_j, d_ij (rank-one updates of M^-1).
from numba import njit

@njit(cache=False)
def _mv(A, v):
    n = A.shape[0]; out = np.zeros(n)
    for a in range(n):
        acc = 0.0
        for b in range(n): acc += A[a, b]*v[b]
        out[a] = acc
    return out

@njit(cache=False)
def _dot(u, v):
    acc = 0.0
    for a in range(u.shape[0]): acc += u[a]*v[a]
    return acc

@njit(cache=False)
def _exchange(Hs, xi, W, cap, tol, max_sweeps, batch):
    K_, p = Hs.shape
    finite_cap = np.isfinite(cap)
    gap = np.inf; sweeps = 0; n_ex = 0
    d = np.empty(K_)
    while sweeps < max_sweeps:
        M = np.zeros((p, p))
        for k in range(K_):
            if xi[k] > 0.0:
                for a in range(p):
                    for b in range(p): M[a, b] += xi[k]*Hs[k, a]*Hs[k, b]
        Mi = np.linalg.inv(M)
        for k in range(K_):
            acc = 0.0
            for a in range(p):
                ta = 0.0
                for b in range(p): ta += Mi[a, b]*Hs[k, b]
                acc += Hs[k, a]*ta
            d[k] = acc
        # certificate: top-W sum of d over the feasible set
        order = np.argsort(-d)
        if finite_cap:
            n = int(np.floor(W/cap)); S = 0.0
            for r in range(n): S += cap*d[order[r]]
            if n < K_: S += (W - n*cap)*d[order[n]]
        else:
            S = W*d[order[0]]
        xd = 0.0
        for k in range(K_): xd += xi[k]*d[k]
        gap = S - xd                                       # cutting-plane gap
        if gap <= tol: break
        sweeps += 1
        # candidate lists
        up = np.empty(batch, np.int64); nu_ = 0
        for r in range(K_):
            k = order[r]
            if (not finite_cap) or xi[k] < cap:
                up[nu_] = k; nu_ += 1
                if nu_ == batch: break
        dn = np.empty(batch, np.int64); nd_ = 0
        for r in range(K_-1, -1, -1):
            k = order[r]
            if xi[k] > 0.0:
                dn[nd_] = k; nd_ += 1
                if nd_ == batch: break
        nb = min(nu_, nd_)
        for r in range(nb):
            i = up[r]; j = dn[r]
            if i == j: continue
            ui = _mv(Mi, Hs[i]); uj = _mv(Mi, Hs[j])
            di = _dot(Hs[i], ui); dj = _dot(Hs[j], uj); dij = _dot(Hs[i], uj)
            if di <= dj: continue
            den = 2.0*(di*dj - dij*dij)
            dmax = xi[j]
            if finite_cap and cap - xi[i] < dmax: dmax = cap - xi[i]
            dl = dmax if den <= 0.0 else min((di - dj)/den, dmax)
            if dl <= 0.0: continue
            # rank-one updates: + dl h_i h_i^T, then - dl h_j h_j^T
            Mi = Mi - (dl/(1.0 + dl*di))*np.outer(ui, ui)
            uj = _mv(Mi, Hs[j]); djn = _dot(Hs[j], uj)
            Mi = Mi + (dl/(1.0 - dl*djn))*np.outer(uj, uj)
            xi[i] += dl; xi[j] -= dl
            if xi[j] < 1e-15: xi[j] = 0.0
            if finite_cap and xi[i] > cap - 1e-15: xi[i] = cap
            n_ex += 1
    return xi, gap, sweeps, n_ex

def exchange(Hs, W, cap, xi0=None, tol=1e-10, max_sweeps=200000, batch=64):
    """maximise log det(sigma^-2 sum xi_k h_k h_k^T) over {0 <= xi <= cap, sum xi = W}; stops when the cutting-plane gap <= tol."""
    K_ = Hs.shape[0]
    xi = proj_capped(np.full(K_, W/K_) if xi0 is None else np.asarray(xi0, float), W, cap).copy()
    xi, gap, sw, nex = _exchange(np.ascontiguousarray(Hs), xi, float(W), float(cap), float(tol), int(max_sweeps), int(batch))
    # restore sum W after floating-point drift (|drift| ~ 1e-12) on the interior weights only, so exact zeros and caps stay exact
    inter = (xi > 0) & (xi < cap) if np.isfinite(cap) else (xi > 0)
    if inter.any(): xi[inter] += (W - xi.sum())/inter.sum()
    xi = np.clip(xi, 0.0, cap)
    f, ub, gap = certificate(Hs, xi, W, cap)
    return dict(xi=xi, f=f, ub=ub, gap=gap, sweeps=int(sw), n_ex=int(nex))


# --- convex solver ------------------------------------------------------------------------------------------------------
def cvx_solve(Hs, W, cap, solver='CLARABEL', **kw):
    import cvxpy as cp
    K_ = Hs.shape[0]
    A = np.einsum('ki,kj->ijk', Hs, Hs).reshape(P*P, K_)  # vec(M) = A xi
    xi = cp.Variable(K_)
    M = cp.reshape(A@xi, (P, P), order='C')
    cons = [xi >= 0, cp.sum(xi) == W] + ([xi <= cap] if np.isfinite(cap) else [])
    prob = cp.Problem(cp.Maximize(cp.log_det(M)), cons)
    t0 = time.time(); prob.solve(solver=solver, **kw); el = time.time()-t0
    x_raw = np.asarray(xi.value, float)
    x = proj_capped(x_raw, W, cap)                        # remove solver-tolerance infeasibility before certifying
    return dict(xi=x, xi_raw=x_raw, proj_dist=float(np.abs(x-x_raw).max()), status=prob.status,
                obj=float(prob.value - P*LOG_S2), time=el, stats=prob.solver_stats)


def blocks(mask, tt):
    """contiguous runs of True in mask: list of (t_start, t_end, n_points)."""
    m = np.concatenate([[False], mask, [False]]).astype(int); e = np.flatnonzero(np.diff(m))
    return [(float(tt[a]), float(tt[b-1]), int(b-a)) for a, b in zip(e[::2], e[1::2])]
