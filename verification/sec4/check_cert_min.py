#!/usr/bin/env python3
"""Independent check of Section IV-B (upper bound for SoD-class triggers).

Own implementation (no code from the repository) of
  - the backward recursion for the additive objective  sum_k E[gamma_k c_k - lambda q_k]
  - the forward evaluation of the resulting trigger by the density recursion (Proposition 1)
  - the bound  log det J(T) <= -log det A - p + gamma_1 + V*_lambda + lambda (B - 1)
on a grid with step sigma/FAC and a finite candidate set for the two window edges:
  lower edge = min(x_k - e_lo*sigma, r - sigma),  upper edge = max(x_k + e_up*sigma, r + sigma),
  both rounded outward to grid nodes; e = None means "no edge relative to the predicted output".

usage: check_cert_min.py FAC SET [NBIS] [NRND] [MIN]

Variant of check_cert.py with an optional fifth argument MIN: the minimum distance (in units of sigma,
default 1.0, a multiple of 1/FAC) between each window edge and the last transmitted value r, i.e.
  lower edge = min(x_k - e_lo*sigma, r - MIN*sigma),  upper edge = max(x_k + e_up*sigma, r + MIN*sigma).
MIN = 1 is the setting of check_cert.py. The constant-SoD half-width (20 sigma) does not depend on MIN.
"""
import numpy as np, sys, time
from scipy.special import ndtr
from scipy.optimize import brentq

FAC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SET = sys.argv[2] if len(sys.argv) > 2 else 'small'
NBIS = int(sys.argv[3]) if len(sys.argv) > 3 else 7
NRND = int(sys.argv[4]) if len(sys.argv) > 4 else 2
MIN = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
M = int(round(MIN*FAC))                   # minimum edge distance in grid nodes
if M < 1 or abs(MIN*FAC - M) > 1e-9:
    sys.exit(f"MIN = {MIN} must be a positive multiple of 1/FAC = 1/{FAC}")
if SET == 'small':
    E = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
elif SET == 'full':   # the 30 finite candidate values listed in the 9 Nov draft, eq. (14)
    E = sorted(set([0.5, 0.75] + [round(1.0+0.1*i, 1) for i in range(21)] + [3.5, 4, 5, 6] + [1.25, 1.75, 2.25]))
E = np.array(E); nE = len(E) + 1          # last index = None

# ---------------- model ----------------
om = np.array([3*np.sqrt(2), 3*np.pi, 12.0, 12.3])
a0, am = 1.0, 0.5
dt, K = 1e-3, 5000
G = lambda s: 2.0/(1.0+s)**2
theta = np.concatenate([[G(0).real], np.ravel([[G(1j*w).real, G(1j*w).imag] for w in om])])
tt = dt*np.arange(1, K+1)
H = np.ones((K, 9)); H[:, 0] = a0
for i, w in enumerate(om):
    H[:, 1+2*i] = am*np.cos(w*tt); H[:, 2+2*i] = -am*np.sin(w*tt)
x = H @ theta
DELTA = 0.002384694; sig = DELTA/20.0; p = 9
SQ = np.sqrt(2*np.pi)

d = sig/FAC
margin = 14
lo0 = x.min() - margin*sig
n = int(np.ceil((x.max() + margin*sig - lo0)/d)) + 1
nodes = lo0 + d*np.arange(n)
idx = np.arange(n)
print(f"grid sigma/{FAC}: {n} nodes; candidates per side: {nE} ({nE-1} values + None)", flush=True)
print(f"MIN = {MIN:g} sigma ({M} nodes)", flush=True)

def logdet(w):
    J = (H*w[:, None]).T @ H / sig**2
    return np.linalg.slogdet(J)[1], J

def step_arrays(k):
    """padded per-node arrays, index j+1 for node j; index 0 = -inf, index n+1 = +inf"""
    al = (nodes - x[k])/sig
    ph = np.exp(-0.5*al*al)/SQ
    g = ph/ph.sum()
    Phi = ndtr(al)
    pad = lambda v, a, b: np.concatenate([[a], v, [b]])
    cg = np.cumsum(g)
    lq = pad(cg - 0.5*g, 0.0, 1.0)                   # mass strictly below node j + half of node j
    uq = pad((cg[-1] - cg) + 0.5*g, 1.0, 0.0)        # mass strictly above node j + half of node j
    return g, pad(al*ph, 0.0, 0.0), pad(ph, 0.0, 0.0), pad(Phi, 0.0, 1.0), lq, uq

def cq(Ai, Bi, aph, ph, Phi, lq, uq):
    """information fraction (continuous formula at the node positions of the edges) and send probability"""
    Pc = Phi[Bi] - Phi[Ai]
    with np.errstate(divide='ignore', invalid='ignore'):
        c = 1 - Pc - aph[Ai] + aph[Bi] + (ph[Ai] - ph[Bi])**2/Pc
    c = np.where(Pc < 1e-280, 1.0, c)        # window entirely in the far tail: the sample is sent almost surely
    return c, lq[Ai] + uq[Bi]

def edge_idx(k):
    """node indices (padded, +1) of the candidate edges at step k, rounded outward"""
    a = np.floor((x[k] - E*sig - lo0)/d + 1e-9).astype(int) + 1
    b = np.ceil((x[k] + E*sig - lo0)/d - 1e-9).astype(int) + 1
    return a, b

BIG = 10**9
def windows(k, code_lo, code_up):
    """padded edge indices for every node, given the codes of the two edges"""
    a, b = edge_idx(k)
    a = np.concatenate([a, [BIG]]); b = np.concatenate([b, [-BIG]])
    Ai = np.minimum(idx + 1 - M, a[code_lo]); Bi = np.maximum(idx + 1 + M, b[code_up])
    return np.clip(Ai, 0, n+1), np.clip(Bi, 0, n+1)

def backward(gam, lam, store=True):
    V = np.zeros(n)
    CL = np.empty((K, n), dtype=np.int8) if store else None
    CU = np.empty((K, n), dtype=np.int8) if store else None
    for k in range(K-1, 0, -1):
        g, aph, ph, Phi, lq, uq = step_arrays(k)
        gV = g*V; cs = np.cumsum(gV)
        lS = np.concatenate([[0.0], cs - 0.5*gV, [cs[-1]]])
        uS = np.concatenate([[cs[-1]], (cs[-1] - cs) + 0.5*gV, [0.0]])
        a, b = edge_idx(k)
        iL = np.clip(idx + 1 - M, 0, n+1); iU = np.clip(idx + 1 + M, 0, n+1)
        def val(Ai, Bi, Vi):
            c, q = cq(Ai, Bi, aph, ph, Phi, lq, uq)
            return gam[k]*c - lam*q + (1 - q)*Vi + lS[Ai] + uS[Bi]
        i_hi = max(0, min(n, b.max() - M - 1))        # nodes i >= i_hi: upper edge is always r + MIN sigma
        i_lo = max(0, min(n, a.min() + M))            # nodes i <  i_lo: lower edge is always r - MIN sigma
        Vn = np.empty(n); cl = np.empty(n, dtype=np.int8); cu = np.empty(n, dtype=np.int8)
        # region above: choose the lower edge only
        sl = slice(i_hi, n)
        Ai = np.minimum(iL[sl][None, :], np.concatenate([a, [BIG]])[:, None])
        v = val(Ai, np.broadcast_to(iU[sl], Ai.shape), V[sl][None, :])
        j = np.argmax(v, axis=0); Vn[sl] = v[j, np.arange(v.shape[1])]; cl[sl] = j; cu[sl] = nE-1
        # region below: choose the upper edge only
        sl = slice(0, i_lo)
        Bi = np.maximum(iU[sl][None, :], np.concatenate([b, [-BIG]])[:, None])
        v = val(np.broadcast_to(iL[sl], Bi.shape), Bi, V[sl][None, :])
        j = np.argmax(v, axis=0); Vn[sl] = v[j, np.arange(v.shape[1])]; cu[sl] = j; cl[sl] = nE-1
        # middle: both edges
        sl = slice(i_lo, i_hi)
        if i_hi > i_lo:
            Ai = np.minimum(iL[sl][None, None, :], np.concatenate([a, [BIG]])[:, None, None])
            Bi = np.maximum(iU[sl][None, None, :], np.concatenate([b, [-BIG]])[None, :, None])
            Ai, Bi = np.broadcast_arrays(Ai, Bi)
            v = val(Ai, Bi, V[sl][None, None, :]).reshape(nE*nE, -1)
            j = np.argmax(v, axis=0); Vn[sl] = v[j, np.arange(v.shape[1])]; cl[sl] = j // nE; cu[sl] = j % nE
        V = Vn
        if store: CL[k] = cl; CU[k] = cu
    g1 = step_arrays(0)[0]
    return float(g1 @ V), CL, CU

def forward(win):
    """density recursion; win(k) -> padded edge indices (Ai, Bi) for every node. Returns w, E[N], per-step E[q]."""
    m = step_arrays(0)[0].copy()
    w = np.empty(K); w[0] = 1.0; Eq = np.zeros(K); Eq[0] = 1.0
    for k in range(1, K):
        g, aph, ph, Phi, lq, uq = step_arrays(k)
        Ai, Bi = win(k)
        c, q = cq(Ai, Bi, aph, ph, Phi, lq, uq)
        w[k] = m @ c; Eq[k] = m @ q
        # arrivals at node j: mass of nodes whose window excludes j (half weight on the edge nodes)
        mA = np.bincount(Ai, weights=m, minlength=n+2); mB = np.bincount(Bi, weights=m, minlength=n+2)
        cA = np.cumsum(mA); cB = np.cumsum(mB)
        T = (cA[-1] - cA) + (cB - mB) + 0.5*(mA + mB)      # sum_{A_i > j} m_i + sum_{B_i < j} m_i + halves
        m = m*(1 - q) + g*T[1:n+1]
    return w, Eq.sum(), Eq, m.sum()

def table_win(CL, CU):
    return lambda k: windows(k, CL[k], CU[k])

if __name__ == "__main__":
    t0 = time.time()
    # ---- baseline: constant SoD on this grid ----
    hw = FAC*20
    w0, B, Eq0, mass = forward(lambda k: (np.clip(idx+1-hw, 0, n+1), np.clip(idx+1+hw, 0, n+1)))
    ld0, J0 = logdet(w0)
    print(f"constant SoD: E[N] = {B:.4f}, log det J = {ld0:.4f}, mass {mass:.12f}   ({time.time()-t0:.0f} s)", flush=True)

    # ---- one-parameter rule: both edges e sigma beyond the predicted output (not rounded to candidates) ----
    def rule_win(e):
        def f(k):
            a = int(np.floor((x[k] - e*sig - lo0)/d + 1e-9)) + 1; b = int(np.ceil((x[k] + e*sig - lo0)/d - 1e-9)) + 1
            return np.clip(np.minimum(idx+1-M, a), 0, n+1), np.clip(np.maximum(idx+1+M, b), 0, n+1)
        return f

    results = []
    A = np.linalg.inv(J0); ld_tan = ld0; name = "constant SoD"
    for rnd in range(NRND):
        gam = np.einsum('ki,ij,kj->k', H, A, H)/sig**2
        print(f"\n== tangent point: J of {name} (log det {ld_tan:.4f}); check sum_k w_k gamma_k = p: "
              f"{(w0 if rnd == 0 else w_best) @ gam:.6f} ==", flush=True)
        gbar = gam[1:].mean()
        lo, hi = None, None; lam = 3.0*gbar; best = None; Dmin = np.inf
        for it in range(NBIS + 6):
            Vs, CL, CU = backward(gam, lam)
            w, EN, Eq, mass = forward(table_win(CL, CU))
            fwd = float(gam[1:] @ w[1:] - lam*(EN - 1))
            ld = logdet(w)[0]
            D = gam[0] + Vs + lam*(B - 1)
            bound = ld_tan - p + D
            Dmin = min(Dmin, D)
            print(f"  lambda/gbar = {lam/gbar:8.4f}: E[N] = {EN:9.3f}  log det J = {ld:9.4f}  "
                  f"V* = {Vs:.6f}  forward score = {fwd:.6f} (diff {Vs-fwd:+.1e})  bound = {bound:.4f}  ({time.time()-t0:.0f} s)", flush=True)
            if EN <= B:
                hi = lam
                if best is None or ld > best[0]: best = (ld, EN, w.copy(), lam)
            else:
                lo = lam
            if lo is None: lam = hi*0.7
            elif hi is None: lam = lo*1.5
            else:
                lam = 0.5*(lo + hi)
                if hi/lo < 1.0005: break
        UB = ld_tan - p + Dmin
        ld_best, EN_best, w_best, lam_best = best
        print(f"  best feasible table: log det J = {ld_best:.4f} (+{ld_best-ld0:.4f}), E[N] = {EN_best:.3f}")
        print(f"  upper bound for the class: {UB:.4f} (+{UB-ld0:.4f});  gap = {UB-ld_best:.4f}")
        results.append((UB, ld_best))
        A = np.linalg.inv(logdet(w_best)[1]); ld_tan = ld_best; name = "the table just found"

    UB = min(r[0] for r in results)
    print(f"\n== other SoD-class triggers at the same budget, against the smallest bound {UB:.4f} ==")
    print(f"  constant SoD                      {ld0:.4f}   below bound: {ld0 <= UB}")
    for r in results: print(f"  table                             {r[1]:.4f}   below bound: {r[1] <= UB}")
    e_star = brentq(lambda e: forward(rule_win(e))[1] - B, 1.2, 2.6, xtol=1e-4)
    for e in (e_star + 1e-4, ):
        w, EN, _, _ = forward(rule_win(e)); ld = logdet(w)[0]
        print(f"  one-parameter rule, e = {e:.4f}:   {ld:.4f} (+{ld-ld0:.4f}), E[N] = {EN:.3f}   below bound: {ld <= UB}")
    print(f"time {time.time()-t0:.0f} s")
