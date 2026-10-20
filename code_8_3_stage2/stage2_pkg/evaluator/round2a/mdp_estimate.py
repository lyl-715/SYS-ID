"""Round 2a, B3/B4: discrete closed-loop MDP — a numerical estimate, NOT a bound (spec reports_CN/plan_recheck_bounds_CN.md v4.2,
§B3, §B4). Labelled "离散 MDP 对偶估算值" everywhere.

Model (units of sigma; Delta/sigma = 20 by the scenario's definition). State delta = r - x_k on the grid
delta_i/sigma = -60 + i/24, i = 0..2880 (delta in [-Delta - 40 sigma, Delta + 40 sigma], h = sigma/24); states beyond the ends map
to the end nodes. Actions v = l h, l = -110..110 (L = 110 kept as the spec fixes it). Per state: c_i = mc83.cfun(alpha_i, beta_i)
(cev + cno), q_i = Phi(alpha_i) + Phi(-beta_i) (= 1 - P_i, written without cancellation), P_i := 1 - q_i (so every row of the
transition sums to one). Transition from i under v: with prob P_i to i - l; with prob q_i to j - l, where the noise e = z_j has the
Gaussian discretised to the grid, phat_j ~ phi(z_j/sigma), masked by |z_j - delta_i| >= Delta (|j - i| >= M = 480) and renormalised
per row: psi_ij = phat_j 1{|j-i| >= M} / N_i. The masked sums are left prefix + right suffix sums (the spec's "total minus window"
as the same quantity; total minus window cancels to 0 in float64 near delta = 0, where N_i ~ 1e-88).
Value iteration backwards over k = K..2 with reward c - lambda q, V_K = c - lambda q, V_k = r + max_v E[V_{k+1} | i, v];
initial state delta_2 = e_1 - v_1 with e_1 unconditional: V~_lambda = max_{v_1} sum_j phat_j V_2(z_j - v_1).
Outer: lambda in [0, 20], golden section to bracket width 1e-3; S~_dual = 1 + min_lambda {V~_lambda + 315.892 lambda}
(315.892 = numerical B - 1, spec v4.2). Ties in max_v are broken towards the smallest |v| (then +v before -v).
Policy at the minimising lambda: forward propagation on the same discrete model gives its own S and E[N].
B4 consistency check first (must pass before any estimate is reported): the uniform input's increments x_{k+1} - x_k rounded to
multiples of h, forward propagation vs fixed-policy backward evaluation of S and E[N], relative 1e-10; diagnostic vs the
evaluator's uniform-input S and E[N] (no pass line).
Writes logs/mdp_estimate.log (one line per number), results/mdp_policy.npz. Usage: nohup python3 mdp_estimate.py > logs/mdp.stdout &"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); sys.path.insert(0, EV)
import numpy as np
from numba import njit
import density_recursion_v1 as dr

N_SIG = 24; HALF = 60; G = 2*HALF*N_SIG+1; M = 20*N_SIG; L = 110
LAM_BUDGET = 315.892                                   # B - 1 (spec v4.2 B3)
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
ORDER = np.array([0]+[s*l for l in range(1, L+1) for s in (1, -1)], np.int64)     # action order: smallest |v| first


def model():
    t = -HALF+np.arange(G)/N_SIG                        # delta/sigma
    al, be = t-20.0, t+20.0
    cev, cno = dr.cfun(al, be); c = cev+cno
    q = dr.ndtr(al)+dr.ndtr(-be); P = 1.0-q
    g = np.exp(-0.5*t*t); phat = g/g.sum()
    Cl = np.concatenate([[0.0], np.cumsum(phat)])       # left prefix
    Sr = np.concatenate([np.cumsum(phat[::-1])[::-1], [0.0]])   # right suffix
    N = np.array([(Cl[i-M+1] if i-M >= 0 else 0.0)+(Sr[i+M] if i+M <= G-1 else 0.0) for i in range(G)])
    return t, c, q, P, phat, 1.0/N, N


@njit(cache=True)
def _tails(A, Cl, Sr):
    n = A.size; Cl[0] = 0.0
    for j in range(n): Cl[j+1] = Cl[j]+A[j]
    Sr[n] = 0.0
    for j in range(n-1, -1, -1): Sr[j] = Sr[j+1]+A[j]

@njit(cache=True)
def _clip(i, n):
    return 0 if i < 0 else (n-1 if i > n-1 else i)

@njit(cache=True)
def _q_action(Vn, P, q, phat, Ninv, M, l, A, Cl, Sr, out):
    """out_i = P_i Vn(i - l) + q_i sum_{|j-i|>=M} phat_j Vn(j - l) / N_i."""
    n = Vn.size
    for j in range(n): A[j] = phat[j]*Vn[_clip(j-l, n)]
    _tails(A, Cl, Sr)
    for i in range(n):
        s = (Cl[i-M+1] if i-M >= 0 else 0.0)+(Sr[i+M] if i+M <= n-1 else 0.0)
        out[i] = P[i]*Vn[_clip(i-l, n)]+q[i]*s*Ninv[i]

@njit(cache=True)
def _max_step(Vn, r, P, q, phat, Ninv, M, order, Vout, pol):
    n = Vn.size; A = np.empty(n); Cl = np.empty(n+1); Sr = np.empty(n+1); tmp = np.empty(n)
    for i in range(n): Vout[i] = -np.inf
    for a in range(order.size):
        l = order[a]
        _q_action(Vn, P, q, phat, Ninv, M, l, A, Cl, Sr, tmp)
        for i in range(n):
            if tmp[i] > Vout[i]: Vout[i] = tmp[i]; pol[i] = l
    for i in range(n): Vout[i] += r[i]

@njit(cache=True)
def _init_value(V2, phat, order):
    """max over v_1 of sum_j phat_j V_2(j - l), and the maximiser."""
    n = V2.size; best = -np.inf; bl = 0
    for a in range(order.size):
        l = order[a]; s = 0.0
        for j in range(n): s += phat[j]*V2[_clip(j-l, n)]
        if s > best: best = s; bl = l
    return best, bl

@njit(cache=True)
def _forward_step(mu, P, q, phat, Ninv, M, pol, used, out):
    """out = mu transported one step under the state-dependent action pol (index shifts); used[l+L] flags the actions present."""
    n = mu.size; A = np.empty(n); Cl = np.empty(n+1); Sr = np.empty(n+1); Lh = (used.size-1)//2
    for i in range(n): out[i] = 0.0
    for i in range(n): out[_clip(i-pol[i], n)] += mu[i]*P[i]
    for li in range(used.size):
        if not used[li]: continue
        l = li-Lh
        for i in range(n): A[i] = mu[i]*q[i]*Ninv[i] if pol[i] == l else 0.0
        _tails(A, Cl, Sr)
        for j in range(n):
            s = (Cl[j-M+1] if j-M >= 0 else 0.0)+(Sr[j+M] if j+M <= n-1 else 0.0)
            out[_clip(j-l, n)] += phat[j]*s


def initial_mu(phat, l1):
    mu = np.zeros(G)
    for j in range(G): mu[min(max(j-l1, 0), G-1)] += phat[j]
    return mu

def forward(pols, l1, c, q, P, phat, Ninv):
    """S and E[N] of a policy: pols[k] = array of index shifts at step k (k = 2..K-1 stored at index k), l1 the initial shift."""
    mu = initial_mu(phat, l1); S = 1.0; EN = 1.0; out = np.empty(G); used = np.zeros(2*L+1, np.bool_)
    for k in range(2, dr.K+1):
        S += float(mu@c); EN += float(mu@q)
        if k < dr.K:
            pk = pols(k); used[:] = False; used[np.unique(pk)+L] = True
            _forward_step(mu, P, q, phat, Ninv, M, pk, used, out); mu, out = out, mu
    return S, EN, float(mu.sum())

def backward_fixed(lseq, l1, rew, P, q, phat, Ninv):
    """fixed open-loop policy (one shift per step): value of sum_{k>=2} E[rew], read at the initial state."""
    V = rew.copy(); A = np.empty(G); Cl = np.empty(G+1); Sr = np.empty(G+1); tmp = np.empty(G)
    for k in range(dr.K-1, 1, -1):
        _q_action(V, P, q, phat, Ninv, M, lseq[k], A, Cl, Sr, tmp); V = rew+tmp
    return float(sum(phat[j]*V[min(max(j-l1, 0), G-1)] for j in range(G)))

def dual_value(lam, c, q, P, phat, Ninv, store=False):
    r = c-lam*q; V = r.copy(); Vout = np.empty(G); pol = np.zeros(G, np.int64)
    pols = np.zeros((dr.K+1, G), np.int16) if store else None
    for k in range(dr.K-1, 1, -1):
        _max_step(V, r, P, q, phat, Ninv, M, ORDER, Vout, pol); V, Vout = Vout, V
        if store: pols[k] = pol
    Vt, l1 = _init_value(V, phat, ORDER)
    return Vt, l1, pols


if __name__ == '__main__':
    os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)
    sys.stdout = dr.Tee(os.path.join(LOGS, 'mdp_estimate.log'))
    t0 = time.time()
    t, c, q, P, phat, Ninv, N = model()
    print(f"=== mdp_estimate.py (round 2a B3/B4; 离散 MDP 对偶估算值, not a bound); grid G = {G} (delta/sigma in [-60, 60], h = sigma/{N_SIG}), "
          f"M = {M}, actions l = -{L}..{L} ({L} h = {L/N_SIG:.4f} sigma), K = {dr.K}, Lagrangian term {LAM_BUDGET} lambda")
    print(f"MDP min N_i {float(N.min())!r} (masked Gaussian mass, all > 0: {bool((N > 0).all())}); row sums of the send kernel are 1 by construction")
    # --- B4 consistency check (must pass first) ------------------------------------------------------------------------
    xU, _ = dr.input_arrays('U'); hsig = dr.s/N_SIG
    lseq_full = np.rint(np.diff(xU)/hsig).astype(np.int64)            # v_k for k = 1..K-1 at index k-1
    l1 = int(lseq_full[0]); lseq = np.zeros(dr.K+1, np.int64); lseq[2:dr.K] = lseq_full[1:]   # lseq[k] = v_k, k = 2..K-1
    print(f"B4 open-loop policy: uniform input's x_(k+1) - x_k rounded to multiples of h: |l| max {int(np.abs(lseq_full).max())} "
          f"(<= L = {L}: {bool(np.abs(lseq_full).max() <= L)}); max rounding error {float(np.abs(np.diff(xU)/hsig-lseq_full).max())!r} h")
    tb = time.time()
    Sf, ENf, mass = forward(lambda k: np.full(G, lseq[k], np.int64), l1, c, q, P, phat, Ninv)
    Sb = 1.0+backward_fixed(lseq, l1, c, P, q, phat, Ninv); ENb = 1.0+backward_fixed(lseq, l1, q, P, q, phat, Ninv)
    rS, rN = abs(Sf-Sb)/abs(Sb), abs(ENf-ENb)/abs(ENb)
    ok = rS <= 1e-10 and rN <= 1e-10
    print(f"B4 forward S {Sf!r} E[N] {ENf!r} (final mass {mass!r}); backward S {Sb!r} E[N] {ENb!r}")
    print(f"B4 relative difference S {rS!r}, E[N] {rN!r}; <= 1e-10: {ok}; time {time.time()-tb:.1f} s")
    recU = json.load(open(os.path.join(EV, 'results', 'rec_B_U_n24_m8.json')))
    print(f"B4 diagnostic (no pass line): forward vs evaluator uniform input (B, n = 24): S {Sf!r} vs {recU['S']!r} "
          f"(difference {Sf-recU['S']!r}, relative {(Sf-recU['S'])/recU['S']!r}); E[N] {ENf!r} vs {recU['EN']!r} "
          f"(difference {ENf-recU['EN']!r}, relative {(ENf-recU['EN'])/recU['EN']!r})")
    if not ok:
        print("B4 FAILED: no estimate is reported (spec §B4)"); sys.exit(1)
    # --- golden section over lambda ---------------------------------------------------------------------------------------
    evals = {}
    def F(lam):
        lam = float(lam); ts = time.time(); Vt, l1_, _ = dual_value(lam, c, q, P, phat, Ninv)
        evals[lam] = (float(Vt), int(l1_)); f = float(Vt+LAM_BUDGET*lam)
        print(f"GOLD lambda {lam!r} V~ {Vt!r} V~ + {LAM_BUDGET} lambda {f!r} (v_1 = {l1_} h; {time.time()-ts:.1f} s)")
        return f
    gr = float((np.sqrt(5.0)-1)/2); lo, hi = 0.0, 20.0
    cc = hi-gr*(hi-lo); dd = lo+gr*(hi-lo); Fc = F(cc); Fd = F(dd)
    while hi-lo > 1e-3:
        if Fc < Fd: hi, dd, Fd = dd, cc, Fc; cc = hi-gr*(hi-lo); Fc = F(cc)
        else: lo, cc, Fc = cc, dd, Fd; dd = lo+gr*(hi-lo); Fd = F(dd)
    lam_s = min(evals, key=lambda k: evals[k][0]+LAM_BUDGET*k); Fmin = evals[lam_s][0]+LAM_BUDGET*lam_s
    print(f"GOLD final bracket [{lo!r}, {hi!r}] (width {hi-lo!r}); {len(evals)} evaluations")
    print(f"MDP 离散 MDP 对偶估算值 S~_dual = 1 + min_lambda {{V~ + {LAM_BUDGET} lambda}} = {1+Fmin!r} at lambda* = {lam_s!r} "
          f"(V~ = {evals[lam_s][0]!r}); an estimate on the discrete model, not a bound")
    # --- policy at lambda*: store, forward ------------------------------------------------------------------------------
    Vt, l1s, pols = dual_value(lam_s, c, q, P, phat, Ninv, store=True)
    Sp, ENp, massp = forward(lambda k: pols[k].astype(np.int64), l1s, c, q, P, phat, Ninv)
    print(f"POLICY at lambda* = {lam_s!r}: forward on the discrete model S {Sp!r}, E[N] {ENp!r} (final mass {massp!r}); v_1 = {l1s} h")
    print(f"POLICY E[N] <= B = 316.8921168493721: {ENp <= 316.8921168493721}; E[N] <= 316: {ENp <= 316}; "
          f"E[N] - 316 = {ENp-316!r} ({(ENp-316)/316*100!r} %); conclusions limited to the discrete model")
    np.savez_compressed(os.path.join(RES, 'mdp_policy.npz'), t=t, pols=pols, l1=l1s, lam=lam_s, c=c, q=q)
    print(f"MDP total runtime {time.time()-t0:.1f} s")
