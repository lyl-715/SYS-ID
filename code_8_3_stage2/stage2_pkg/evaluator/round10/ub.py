"""Round 10: nominal optimum of log det J at a given budget over the window class W, with a certified gap
(spec reports_CN/plan_round10_logdet_upper_bound_EN.md, frozen 2026-11-08; execution prompt
reports_CN/prompt_round10_claude_code_EN.md §2). numpy/scipy only; no dependence on the evaluator.

    UB(B) = max_q f(q),   f(q) = log det( sigma^-2 [ h_1 h_1^T + sum_{k>=2} c(q_k) h_k h_k^T ] ),
            C = { 0 <= q_k <= 1, sum_{k>=2} q_k <= B - 1 }

  c = c_max (symmetric window, the bound; Lemmas A and B of the plan) or c = c_one (ideal one-sided window, V_1; reference
  only). f is concave on [0, 1]^(K-1) because log det is concave and Loewner-monotone and c is concave and increasing.

Certification (plan §2): at an interior point q_hat (0 < q_hat_k < 1), the tangent plane of the concave f gives, for every
v in C, f(v) <= f(q_hat) + grad f(q_hat)^T (v - q_hat); hence

    f(q_hat) <= UB <= U_cert(q_hat) = f(q_hat) + max_{v in C} grad f(q_hat)^T (v - q_hat),

the left inequality because q_hat is feasible. The inner maximum is a linear budget allocation over the ORIGINAL C (sort
the gradient in descending order, v_k = 1 until the budget is spent, fractional last component, v_k = 0 for non-positive
gradient components): exact, no truncated gradient, no shrunken C.

Stand-alone use (any system):  python3 ub.py --H H.npy --sigma S --B B [--tol 0.01] [--out prefix]
  H.npy: (K, p) array of regressors h_k (row 0 = h_1, always transmitted); prints f(q_hat), U_cert, gap, lambda_hat and
  writes <prefix>_q.npy (q_hat for k = 2..K) and <prefix>.json.
"""
import json, time, argparse
import numpy as np
from scipy.stats import norm
from scipy.special import erfcx
from scipy.linalg import cholesky, solve_triangular

SQ2PI = np.sqrt(2.0/np.pi)


# --- information weights at fixed transmission probability q (units of sigma) ------------------------------------------
def a_sym(q):
    """symmetric half-width a = Phi^{-1}(1 - q/2), computed as norm.isf(q/2) (stable as q -> 0)."""
    return norm.isf(np.asarray(q, float)/2)

def c_max(q):
    """c_max(q) = q + 2 a phi(a): information weight of the symmetric window (-a, a) at send probability q.
    c_max(0) = 0 and c_max(1) = 1 explicitly."""
    q = np.asarray(q, float)
    with np.errstate(invalid='ignore', over='ignore'):
        a = norm.isf(q/2); v = q+2*a*norm.pdf(a)
    return np.where(q <= 0, 0.0, np.where(q >= 1, 1.0, v))

def dc_max(q):
    """dc_max/dq = a^2 (Lemma B); +inf at q = 0, 0 at q = 1."""
    q = np.asarray(q, float)
    a = norm.isf(q/2)
    return np.where(q <= 0, np.inf, np.where(q >= 1, 0.0, a*a))

def a_one(q):
    """one-sided window (alpha, +inf), send iff z <= alpha: alpha = Phi^{-1}(q)."""
    return norm.ppf(np.asarray(q, float))

def c_one(q):
    """c_one(q) = q - alpha phi(alpha) + phi(alpha)^2 / (1 - q), alpha = Phi^{-1}(q); c_one(0) = 0, c_one(1) = 1."""
    q = np.asarray(q, float)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        al = norm.ppf(q); pa = norm.pdf(al)
        v = q-al*pa+pa*pa/(1-q)
    return np.where(q <= 0, 0.0, np.where(q >= 1, 1.0, v))

def dc_one(q):
    """dc_one/dq = (alpha - mu)^2, mu = phi(alpha)/(1 - q) (inverse Mills ratio); +inf at q = 0, 0 at q = 1."""
    q = np.asarray(q, float)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        al = norm.ppf(q); mu = norm.pdf(al)/(1-q)
        v = (al-mu)**2
    return np.where(q <= 0, np.inf, np.where(q >= 1, 0.0, v))

WEIGHTS = {'sym': (c_max, dc_max), 'one': (c_one, dc_one)}


# --- objective and gradient --------------------------------------------------------------------------------------------
class Problem:
    """f(q) and its gradient for regressors Hk (K x p; row 0 = h_1 with weight 1), noise sd sigma, weight family c, dc."""
    def __init__(self, Hk, sigma, c=c_max, dc=dc_max):
        self.Hk = np.ascontiguousarray(Hk, float); self.h1 = self.Hk[0]; self.H = self.Hk[1:]
        self.s2 = float(sigma)**-2; self.c, self.dc = c, dc
        self.n = self.H.shape[0]; self.p = self.Hk.shape[1]
        self.H0 = np.outer(self.h1, self.h1)
        self.nf = 0; self.ng = 0

    def J(self, q, w=None):
        w = self.c(q) if w is None else w
        return self.s2*(self.H0+(self.H*w[:, None]).T@self.H)

    def chol(self, q, w=None):
        """lower Cholesky factor of J(q); raises numpy/scipy LinAlgError if J is not positive definite."""
        return cholesky(self.J(q, w), lower=True)

    def f(self, q, L=None):
        self.nf += 1
        if L is None: L = self.chol(q)
        return float(2*np.sum(np.log(np.diag(L))))

    def gamma(self, L):
        """gamma_k = sigma^-2 h_k^T J^{-1} h_k for k = 2..K (triangular solve with the Cholesky factor)."""
        Y = solve_triangular(L, self.H.T, lower=True)
        return self.s2*np.sum(Y*Y, axis=0)

    def fg(self, q):
        """f(q), grad f(q), gamma (all at one Cholesky factorisation)."""
        self.ng += 1
        L = self.chol(q); gam = self.gamma(L)
        return self.f(q, L), gam*self.dc(q), gam


# --- feasible set C = {0 <= q <= 1, sum q <= Bm1} --------------------------------------------------------------------
def project(y, Bm1, iters=200):
    """Euclidean projection onto C: clip(y - t, 0, 1) with t = 0 if sum clip(y, 0, 1) <= Bm1, else t > 0 found by bisection
    so that sum = Bm1; the upper end of the final bracket is returned, so the result satisfies sum <= Bm1 (feasible)."""
    y = np.asarray(y, float); x = np.clip(y, 0.0, 1.0)
    if x.sum() <= Bm1: return x, 0.0
    lo, hi = 0.0, float(y.max())                 # sum clip(y - 0) > Bm1 >= sum clip(y - max y) = 0
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        if mid <= lo or mid >= hi: break
        if np.clip(y-mid, 0.0, 1.0).sum() > Bm1: lo = mid
        else: hi = mid
    return np.clip(y-hi, 0.0, 1.0), hi

def lin_max(g, Bm1):
    """argmax_{v in C} g^T v (exact): components in descending order of g, v_k = 1 until the budget Bm1 is spent, the last
    one fractional, v_k = 0 for g_k <= 0 and for the rest. Returns v."""
    g = np.asarray(g, float); n = g.size; v = np.zeros(n)
    order = np.argsort(-g, kind='stable'); pos = order[g[order] > 0]
    nfull = int(min(np.floor(Bm1), pos.size)); v[pos[:nfull]] = 1.0
    rem = Bm1-nfull
    if nfull < pos.size and rem > 0: v[pos[nfull]] = min(rem, 1.0)
    assert np.all(v >= 0) and np.all(v <= 1) and v.sum() <= Bm1*(1+1e-15)
    return v

def u_cert(f, g, q, Bm1):
    """U_cert = f(q) + max_{v in C} g^T (v - q); returns (U_cert, v)."""
    v = lin_max(g, Bm1)
    return float(f+g@v-g@q), v

def lambda_hat(g, q, lo=1e-8):
    """multiplier estimate: median gradient over unsaturated components (lo < q_k < 1 - lo; DECISION 10-4)."""
    m = (q > lo) & (q < 1-lo)
    return float(np.median(g[m])) if m.any() else float('nan')


# --- projected gradient ascent with backtracking (plan §2; DECISION 10-2) -------------------------------------------------
def solve(prob, Bm1, tol=0.01, max_iter=20000, t0=None, s_damp=0.99, armijo=1e-4, log=None, q0=None):
    """Projected gradient ascent on C from q0 = (B-1)/(K-1) (interior, feasible).
    Iteration: y = project(q + t g); d = y - q (an ascent direction, g^T d >= |d|^2/t); q+ = q + s d with s = 1 if y is
    strictly inside (0, 1), else s = s_damp < 1 (damping; q+ is then a convex combination of an interior point and a point of
    C, hence feasible and strictly interior); Armijo backtracking s <- s/2 until f(q+) >= f(q) + armijo s g^T d and
    0 < q+ < 1. The trial step t follows the Barzilai-Borwein rule (alternating BB1/BB2, safeguarded) - a step-size rule of
    projected gradient ascent, not a different method. Every iteration computes gap = U_cert - f; stop at gap <= tol.
    Returns dict(q, f, u_cert, gap, lambda_hat, n_iter, history)."""
    n = prob.n
    q = np.full(n, Bm1/n) if q0 is None else np.array(q0, float)
    assert np.all(q > 0) and np.all(q < 1) and q.sum() <= Bm1*(1+1e-14)
    f, g, gam = prob.fg(q)
    t = (t0 if t0 is not None else 0.1*float(q.mean())/float(np.abs(g).mean()))
    hist = []; t_start = time.perf_counter(); it = 0; n_damped = 0; n_backtrack = 0; n_shrink = 0
    while True:
        U, _ = u_cert(f, g, q, Bm1); gap = U-f
        hist.append((it, f, U, gap, t))
        if log is not None and (it < 5 or it % 50 == 0 or gap <= tol): log(f"  it {it:5d}  f {f!r}  U_cert {U!r}  gap {gap:.3e}  t {t:.3e}")
        if gap <= tol or it >= max_iter: break
        y, _ = project(q+t*g, Bm1)
        d = y-q; gd = float(g@d)
        if gd <= 0:                                   # projected step stationary at this t: shrink t and retry
            t *= 0.1; it += 1; n_shrink += 1; continue
        s = 1.0 if (np.all(y > 0) and np.all(y < 1)) else s_damp
        n_damped += int(s < 1.0)
        while True:
            qn = q+s*d
            if np.all(qn > 0) and np.all(qn < 1):
                try:
                    fn, gn, gamn = prob.fg(qn)
                    if fn >= f+armijo*s*gd: break
                except np.linalg.LinAlgError:
                    pass
            s *= 0.5; n_backtrack += 1
            if s < 1e-20: raise RuntimeError("backtracking failed (s < 1e-20)")
        sk = qn-q; yk = gn-g; sy = -float(sk@yk)  # concave f: -s^T y >= 0
        if sy > 0:
            t = float(sk@sk)/sy if it % 2 == 0 else sy/float(yk@yk)
            t = min(max(t, 1e-12), 1e12)
        else:
            t *= 2.0
        q, f, g, gam = qn, fn, gn, gamn; it += 1
    tot = float(q.sum())
    assert np.all(q > 0) and np.all(q < 1) and tot <= Bm1*(1+1e-14), (q.min(), q.max(), tot, Bm1)
    return dict(q=q, f=f, u_cert=U, gap=gap, lambda_hat=lambda_hat(g, q), n_iter=it, history=hist, grad=g, gamma=gam,
                sum_q=tot, wall_s=time.perf_counter()-t_start, n_f=prob.nf, n_g=prob.ng, n_damped=n_damped, n_backtrack=n_backtrack,
                n_t_shrink=n_shrink)


# --- KKT water-filling fixed point (cross-check; DECISION 10-1) -----------------------------------------------------------
def _inv_dc_sym(r2):
    """q with dc_max(q) = a(q)^2 = r2 (r2 >= 0): a = sqrt(r2), q = 2 Phi_bar(a)."""
    return 2*norm.sf(np.sqrt(r2))

def _inv_dc_one(r2):
    """q with dc_one(q) = (mu(alpha) - alpha)^2 = r2: mu - alpha is decreasing in alpha (from +inf to 0); bisection on alpha
    in [-60, 60] with the stable inverse Mills ratio mu = sqrt(2/pi)/erfcx(alpha/sqrt 2); q = Phi(alpha)."""
    r = np.sqrt(r2); lo = np.full(r.shape, -60.0); hi = np.full(r.shape, 60.0)
    for _ in range(80):                                   # 120 / 2^80: below double resolution of alpha
        mid = 0.5*(lo+hi); m = SQ2PI/erfcx(mid/np.sqrt(2))-mid
        big = m > r; lo = np.where(big, mid, lo); hi = np.where(big, hi, mid)
    return norm.cdf(0.5*(lo+hi))

INV_DC = {'sym': _inv_dc_sym, 'one': _inv_dc_one}

def water_fill(gam, Bm1, inv_dc):
    """given gamma_k, the q(lambda) with gamma_k dc(q_k) = lambda (all components interior because dc(0+) = inf and
    dc(1) = 0), lambda by bisection on log lambda so that sum q(lambda) = Bm1 (sum q decreasing in lambda)."""
    lo, hi = -60.0, 20.0
    for _ in range(100):                                  # 80 / 2^100: below double resolution of log lambda
        mid = 0.5*(lo+hi)
        if inv_dc(np.exp(mid)/gam).sum() > Bm1: lo = mid
        else: hi = mid
    lam = np.exp(hi); q = inv_dc(lam/gam)
    return q, float(lam)

def kkt_fixed_point(prob, Bm1, inv_dc, q0=None, omega=0.5, max_iter=2000, tol_q=1e-13, log=None):
    """fixed-point iteration on J: gamma from J(q), water-filling q_new(gamma), damped update q <- (1-omega) q + omega q_new,
    until max |q_new - q| <= tol_q. Returns dict(q, f, lambda, n_iter, dq, u_cert, gap)."""
    n = prob.n; q = np.full(n, Bm1/n) if q0 is None else np.array(q0, float)
    for it in range(1, max_iter+1):
        L = prob.chol(q); gam = prob.gamma(L)
        qn, lam = water_fill(gam, Bm1, inv_dc)
        dq = float(np.abs(qn-q).max())
        q = (1-omega)*q+omega*qn
        if log is not None and (it <= 3 or it % 100 == 0): log(f"  kkt it {it:4d}  max|dq| {dq:.3e}  lambda {lam!r}")
        if dq <= tol_q: break
    # q is feasible: a convex combination of feasible points (sum <= Bm1 by the bisection's upper end); rescale guard only
    if q.sum() > Bm1: q *= Bm1/q.sum()
    f, g, gam = prob.fg(q); U, _ = u_cert(f, g, q, Bm1)
    return dict(q=q, f=f, lam=lam, n_iter=it, dq=dq, u_cert=U, gap=U-f, sum_q=float(q.sum()))


# --- stand-alone entry point ------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--H', required=True); ap.add_argument('--sigma', type=float, required=True)
    ap.add_argument('--B', type=float, required=True); ap.add_argument('--tol', type=float, default=0.01)
    ap.add_argument('--weights', choices=('sym', 'one'), default='sym'); ap.add_argument('--out', default='ub')
    a = ap.parse_args()
    Hk = np.load(a.H); c, dc = WEIGHTS[a.weights]
    prob = Problem(Hk, a.sigma, c, dc)
    r = solve(prob, a.B-1, tol=a.tol, log=print)
    out = {k: r[k] for k in ('f', 'u_cert', 'gap', 'lambda_hat', 'n_iter', 'sum_q', 'wall_s')}
    out.update(B=a.B, sigma=a.sigma, K=int(Hk.shape[0]), p=int(Hk.shape[1]), weights=a.weights)
    print(json.dumps(out, indent=1))
    np.save(a.out+'_q.npy', r['q']); json.dump(out, open(a.out+'.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
