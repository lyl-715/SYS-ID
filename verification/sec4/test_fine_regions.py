#!/usr/bin/env python3
"""test_fine_regions.py FAC SET MIN  -- checks of check_cert_fine.py's region decomposition for a candidate set
(meant for SET = fine, whose e-values include negative ones).

1. int8: the largest code (nE - 1) fits int8.
2. For every step k = 1..K-1: a.min() (smallest lower-edge index) and b.max() (largest upper-edge index) over the
   candidates are the edges of the largest e, so i_lo = a.min() + M and i_hi = b.max() - M - 1 are what the
   region logic of backward() needs. Also: every negative-e lower edge lies above x_k and every negative-e upper
   edge below x_k (near side).
3. Brute force: for several steps k0, backward() is run on a two-step problem (module K patched to 3, x[1], x[2]
   replaced by x[k0], x[k0+1]) and compared with an exhaustive maximisation over all nE x nE actions at every node
   (the same value expression as backward(), without the three-region split). Reports |difference| of the final
   value g1 @ V and of the per-node maximum at the chosen codes.
"""
import sys, time
import numpy as np
FAC_s, SET_s, MIN_s = sys.argv[1:4]
sys.argv = ['check_cert_fine.py', FAC_s, SET_s, '5', '3', MIN_s]
import check_cert_fine as cc                      # noqa: E402

t0 = time.time()
n, idx, M, E, nE, BIG, K0, x0 = cc.n, cc.idx, cc.M, cc.E, cc.nE, cc.BIG, cc.K, cc.x.copy()
print(f"1. codes per side nE = {nE}; largest code {nE-1} <= 127 (int8): {nE-1 <= np.iinfo(np.int8).max}")

ok_min = ok_max = ok_side = True; jmax = int(np.argmax(E))
for k in range(1, K0):
    a, b = cc.edge_idx(k)
    ok_min &= bool(a.min() == a[jmax]); ok_max &= bool(b.max() == b[jmax])
    neg = E < 0
    if neg.any():
        xi = (x0[k] - cc.lo0)/cc.d + 1                    # padded fractional index of x_k
        ok_side &= bool(np.all(a[neg] > xi - 1) and np.all(b[neg] < xi + 1))
print(f"2. over k = 1..{K0-1}: a.min() is the lower edge of the largest e ({E[jmax]:g}): {ok_min}; "
      f"b.max() is its upper edge: {ok_max}; negative-e edges on the near side of x_k: {ok_side}")

# gamma and lambda as in round 1 of check_cert_min (tangent point: constant SoD)
hw = cc.FAC*20
w0, B, _, _ = cc.forward(lambda k: (np.clip(idx+1-hw, 0, n+1), np.clip(idx+1+hw, 0, n+1)))
A = np.linalg.inv(cc.logdet(w0)[1]); gam_full = np.einsum('ki,ij,kj->k', cc.H, A, cc.H)/cc.sig**2
lam = 3.0*gam_full[1:].mean()

def brute_step(k, V, gk):
    g, aph, ph, Phi, lq, uq = cc.step_arrays(k)
    gV = g*V; cs = np.cumsum(gV)
    lS = np.concatenate([[0.0], cs - 0.5*gV, [cs[-1]]]); uS = np.concatenate([[cs[-1]], (cs[-1] - cs) + 0.5*gV, [0.0]])
    a, b = cc.edge_idx(k); a = np.concatenate([a, [BIG]]); b = np.concatenate([b, [-BIG]])
    iL = np.clip(idx + 1 - M, 0, n+1); iU = np.clip(idx + 1 + M, 0, n+1)
    Vn = np.empty(n); arg = np.empty(n, dtype=np.int64)
    for s in range(0, n, 400):
        sl = slice(s, min(n, s + 400))
        Ai = np.minimum(iL[sl][None, None, :], a[:, None, None]); Bi = np.maximum(iU[sl][None, None, :], b[None, :, None])
        Ai, Bi = np.broadcast_arrays(Ai, Bi)
        c, q = cc.cq(Ai, Bi, aph, ph, Phi, lq, uq)
        v = (gk*c - lam*q + (1 - q)*V[sl][None, None, :] + lS[Ai] + uS[Bi]).reshape(nE*nE, -1)
        arg[sl] = np.argmax(v, axis=0); Vn[sl] = v[arg[sl], np.arange(v.shape[1])]
    return Vn

def value_at(k, V, gk, cl, cu):
    """the value of backward()'s expression at the given codes, every node"""
    g, aph, ph, Phi, lq, uq = cc.step_arrays(k)
    gV = g*V; cs = np.cumsum(gV)
    lS = np.concatenate([[0.0], cs - 0.5*gV, [cs[-1]]]); uS = np.concatenate([[cs[-1]], (cs[-1] - cs) + 0.5*gV, [0.0]])
    a, b = cc.edge_idx(k); a = np.concatenate([a, [BIG]]); b = np.concatenate([b, [-BIG]])
    Ai = np.minimum(np.clip(idx + 1 - M, 0, n+1), a[cl.astype(np.intp)]); Bi = np.maximum(np.clip(idx + 1 + M, 0, n+1), b[cu.astype(np.intp)])
    c, q = cc.cq(Ai, Bi, aph, ph, Phi, lq, uq)
    return gk*c - lam*q + (1 - q)*V + lS[Ai] + uS[Bi]

steps = [int(np.argmin(x0[1:-1])) + 1, int(np.argmax(x0[1:-1])) + 1, 777, 2500, 4321]
worst = 0.0
for k0 in steps:
    xp = x0.copy(); xp[1], xp[2] = x0[k0], x0[k0+1]
    cc.x = xp; cc.K = 3
    gam = np.array([gam_full[0], gam_full[k0], gam_full[k0+1]])
    Vs, CL, CU = cc.backward(gam, lam)                     # region version, steps 2 and 1
    V2 = brute_step(2, np.zeros(n), gam[2]); V1 = brute_step(1, V2, gam[1])
    Vb = float(cc.step_arrays(0)[0] @ V1)
    d2 = np.abs(value_at(2, np.zeros(n), gam[2], CL[2], CU[2]) - V2).max()
    d1 = np.abs(value_at(1, V2, gam[1], CL[1], CU[1]) - V1).max()
    worst = max(worst, abs(Vs - Vb), d1, d2)
    print(f"3. k0 = {k0:4d} (x = {x0[k0]:+.6f}): g1 @ V  region {Vs:.15f}  brute {Vb:.15f}  diff {Vs-Vb:+.1e};  "
          f"max |value at chosen codes - brute max|: step 2 {d2:.1e}, step 1 {d1:.1e}   ({time.time()-t0:.0f} s)", flush=True)
cc.x = x0; cc.K = K0
print(f"worst abs difference over all tested steps: {worst:.1e}")
