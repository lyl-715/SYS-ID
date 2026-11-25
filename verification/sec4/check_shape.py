#!/usr/bin/env python3
"""Shape of the constructed trigger (Task B).

usage: check_shape.py FAC SET MIN RUNFILE [OUT.npz] [MODULE]

Imports MODULE (default: check_cert_fine for SET = fine, else check_cert_min; both unchanged) with arguments
FAC SET 5 3 MIN, then
  1. recomputes the constant-SoD baseline and, round by round, the best feasible table of the run RUNFILE
     (stdout of check_cert.py / check_cert_min.py with the same FAC SET 5 3 MIN). The tangent point of
     round r+1 is J of the best table of round r, so all three rounds are recomputed, with ONE backward
     sweep per round at the price lambda that gave that round's best feasible table. The lambdas are not
     re-searched: the lambda sequence of the run is replayed with the script's own arithmetic
     (3*gbar, x0.7, x1.5, bisection) and the feasibility decisions read from RUNFILE; the best index is the
     feasible entry with the largest log det printed in RUNFILE. Every recomputed number is compared
     with the printed one.
  2. runs one forward pass (a copy of check_cert_min.forward with statistics added) for the round-3 table
     and one for the one-parameter rule at the e printed in RUNFILE's rule line, and collects statistics of
     the two window edges weighted by m, the mass of the last transmitted value r over the grid nodes at
     the start of step k (the array m in forward(), i.e. the weights in w[k] = m @ c), for code steps
     k = 2, ..., K-1 (x[k], k = 0 is the first sample, always sent).

Definitions (all comparisons on integer node indices, padded by +1 as in check_cert_min.py):
  r       = nodes[i] (r is always a grid node).
  facing  = upper edge if r < x_k, lower edge if r > x_k; far = the other edge. r == x_k is counted apart.
  MIN edge of node i: index i+1-M (lower) / i+1+M (upper), M = MIN*FAC; exact, no rounding (r is a node).
  e-edge: candidate e rounded outward as in edge_idx(): floor((x_k - e sig - lo0)/d + 1e-9) + 1 (lower),
          ceil((x_k + e sig - lo0)/d - 1e-9) + 1 (upper); "None" = no e-edge.
  status of an edge: "e"   = the chosen e-edge lies strictly beyond the MIN edge (it sets the edge);
                     "tie" = the chosen e-edge is the MIN node itself (same position either way);
                     "MIN" = code None, or the chosen e-edge lies strictly inside (the MIN edge is used).
  quantile q of a weighted discrete distribution = smallest value v with cumulative weight fraction >= q.
"""
import sys, time, re
import numpy as np

if len(sys.argv) < 5:
    sys.exit(__doc__)
FAC_s, SET_s, MIN_s, RUNFILE = sys.argv[1:5]
OUT = sys.argv[5] if len(sys.argv) > 5 else None
MODULE = sys.argv[6] if len(sys.argv) > 6 else ('check_cert_fine' if SET_s == 'fine' else 'check_cert_min')
sys.argv = [MODULE + '.py', FAC_s, SET_s, '5', '3', MIN_s]
import importlib                                  # noqa: E402
cc = importlib.import_module(MODULE)              # prints the grid and MIN lines
print(f"module: {MODULE}.py", flush=True)

t0 = time.time()
K, n, idx, M, E, nE, BIG = cc.K, cc.n, cc.idx, cc.M, cc.E, cc.nE, cc.BIG
KMIN = 2

# ---------------- parse the run output ----------------
txt = open(RUNFILE).read()
grid_line = txt.splitlines()[0]
assert grid_line.startswith(f"grid sigma/{cc.FAC}: {n} nodes; candidates per side: {nE} "), grid_line
mm = re.search(r"^MIN = ([\d.]+) sigma \((\d+) nodes\)", txt, re.M)
run_M = int(mm.group(2)) if mm else cc.FAC        # check_cert.py has no MIN line: MIN = 1 sigma = FAC nodes
assert run_M == M, f"MIN of {RUNFILE} ({run_M} nodes) differs from MIN = {MIN_s} ({M} nodes)"
m0 = re.search(r"constant SoD: E\[N\] = ([\d.]+), log det J = ([\d.]+)", txt)
rounds_txt = txt.split("== tangent point")[1:]
LAM = re.compile(r"lambda/gbar =\s*([\d.]+): E\[N\] =\s*([\d.]+)\s+log det J =\s*([\d.]+)\s+V\* = ([-\d.]+)")
BEST = re.compile(r"best feasible table: log det J = ([\d.]+) \(\+[\d.]+\), E\[N\] = ([\d.]+)")
runs = []
for rt in rounds_txt[:3]:
    ent = [(a, float(b), float(c), dd) for a, b, c, dd in LAM.findall(rt)]
    bm = BEST.search(rt)
    runs.append((ent, bm.group(1), bm.group(2)))
assert len(runs) == 3, "RUNFILE does not contain three complete rounds"
rule_e_s = re.search(r"one-parameter rule, e = ([\d.]+):\s+([\d.]+) \(\+[\d.]+\), E\[N\] = ([\d.]+)", txt)

# ---------------- statistics ----------------
LOW_NAMES = ("facing e < 0", "facing 0 <= e < 0.5", "far e < 0", "far 0 <= e < 0.5",
             "either edge e < 0", "either edge e < 0.5", "either edge e < 0.5 incl. ties")
def new_hist(nv, values):
    # axes: side (0: r < x_k, 1: r > x_k, 2: r == x_k), near (|r - x_k| < sig), status (e, tie, MIN), value slot
    vals = np.array([np.nan if v is None else v for v in values], dtype=float)
    return {'face': np.zeros((3, 2, 3, nv)), 'far': np.zeros((3, 2, 3, nv)), 'joint': np.zeros((3, 2, 3, 3)),
            'clip': 0.0, 'nv': nv, 'vals': vals, 'low': np.zeros(len(LOW_NAMES))}

def accumulate(Hs, m, k, aE, bE, cl, cu):
    r = cc.nodes; xk = cc.x[k]
    side = np.where(r < xk, 0, np.where(r > xk, 1, 2))
    near = (np.abs(r - xk) < cc.sig).astype(np.intp)
    iLm = idx + 1 - M; iUm = idx + 1 + M
    sL = np.where(aE < iLm, 0, np.where(aE == iLm, 1, 2))
    sU = np.where(bE > iUm, 0, np.where(bE == iUm, 1, 2))
    below = side == 0
    fs = np.where(below, sU, sL); fc = np.where(below, cu, cl)
    rs = np.where(below, sL, sU); rc = np.where(below, cl, cu)
    nv = Hs['nv']; sz = 3*2*3*nv
    Hs['face'] += np.bincount(((side*2 + near)*3 + fs)*nv + fc, weights=m, minlength=sz).reshape(3, 2, 3, nv)
    Hs['far'] += np.bincount(((side*2 + near)*3 + rs)*nv + rc, weights=m, minlength=sz).reshape(3, 2, 3, nv)
    Hs['joint'] += np.bincount(((side*2 + near)*3 + fs)*3 + rs, weights=m, minlength=54).reshape(3, 2, 3, 3)
    lo_ = np.minimum(iLm, aE); hi_ = np.maximum(iUm, bE)
    Hs['clip'] += float(m[(lo_ < 0) | (hi_ > n + 1)].sum())
    # edges set by a small or negative e (status 'e' unless stated; NaN = None compares False)
    fv = Hs['vals'][fc]; rv = Hs['vals'][rc]; ok = side < 2
    fe, re_ = ok & (fs == 0), ok & (rs == 0)
    conds = (fe & (fv < 0), fe & (fv >= 0) & (fv < 0.5), re_ & (rv < 0), re_ & (rv >= 0) & (rv < 0.5),
             (fe & (fv < 0)) | (re_ & (rv < 0)), (fe & (fv < 0.5)) | (re_ & (rv < 0.5)),
             (ok & (fs <= 1) & (fv < 0.5)) | (ok & (rs <= 1) & (rv < 0.5)))
    Hs['low'] += np.array([float(m[c_].sum()) for c_ in conds])

def forward_stats(win, Hs):
    """check_cert_min.forward() line by line, plus accumulate() at steps k >= KMIN.
    win(k) -> (Ai, Bi) clipped padded indices as in forward(), and (aE, bE, cl, cu) for the statistics."""
    m = cc.step_arrays(0)[0].copy()
    w = np.empty(K); w[0] = 1.0; Eq = np.zeros(K); Eq[0] = 1.0
    for k in range(1, K):
        g, aph, ph, Phi, lq, uq = cc.step_arrays(k)
        (Ai, Bi), info = win(k)
        c, q = cc.cq(Ai, Bi, aph, ph, Phi, lq, uq)
        w[k] = m @ c; Eq[k] = m @ q
        if k >= KMIN:
            accumulate(Hs, m, k, *info)
        mA = np.bincount(Ai, weights=m, minlength=n+2); mB = np.bincount(Bi, weights=m, minlength=n+2)
        cA = np.cumsum(mA); cB = np.cumsum(mB)
        T = (cA[-1] - cA) + (cB - mB) + 0.5*(mA + mB)
        m = m*(1 - q) + g*T[1:n+1]
    return w, Eq.sum(), Eq, m.sum()

def table_win_stats(CL, CU):
    def f(k):
        a, b = cc.edge_idx(k)
        a = np.concatenate([a, [BIG]]); b = np.concatenate([b, [-BIG]])
        cl = CL[k].astype(np.intp); cu = CU[k].astype(np.intp)
        Ai, Bi = cc.windows(k, CL[k], CU[k])
        return (Ai, Bi), (a[cl], b[cu], cl, cu)
    return f

def rule_win_stats(e):
    zero = np.zeros(n, dtype=np.intp)
    def f(k):   # rule_win() of check_cert_min.py, copied (it is local to its __main__ block)
        a = int(np.floor((cc.x[k] - e*cc.sig - cc.lo0)/cc.d + 1e-9)) + 1
        b = int(np.ceil((cc.x[k] + e*cc.sig - cc.lo0)/cc.d - 1e-9)) + 1
        Ai = np.clip(np.minimum(idx+1-M, a), 0, n+1); Bi = np.clip(np.maximum(idx+1+M, b), 0, n+1)
        return (Ai, Bi), (np.full(n, a), np.full(n, b), zero, zero)
    return f

def wquant(vals, wts, qs):
    o = np.argsort(vals, kind='stable'); v = vals[o]; c = np.cumsum(wts[o])/wts.sum()
    return [v[np.searchsorted(c, q - 1e-12)] for q in qs]

QS = (0.05, 0.25, 0.50, 0.75, 0.95)
def edge_block(Hx, values, label, sel, denom):
    """Hx[side, near, status, slot] -> printed fractions and e-distribution for the selection sel"""
    h = Hx[sel]                                      # (..., 3, nv) after selecting side/near
    h = h.reshape(-1, 3, h.shape[-1]).sum(axis=0)    # (3, nv)
    tot = h.sum()
    none_slot = len(values) - 1 if values[-1] is None else None
    fe, ft, fm = h[0].sum()/denom, h[1].sum()/denom, h[2].sum()/denom
    out = {'frac_e': fe, 'frac_tie': ft, 'frac_min': fm}
    print(f"    {label}: weight {tot:.4f} (fraction of the selection {tot/denom:.6f})")
    print(f"      set by a candidate e (e-edge strictly beyond the MIN edge): {fe:.4f}")
    print(f"      tie (rounded e-edge = MIN node):                            {ft:.4f}")
    if none_slot is not None:
        print(f"      set by the minimum distance MIN to r:                       {fm:.4f}"
              f"   [code None {h[2, none_slot].sum()/denom:.4f}, e-edge inside {(h[2].sum()-h[2, none_slot])/denom:.4f}]")
    else:
        print(f"      set by the minimum distance MIN to r:                       {fm:.4f}")
    vals = np.array([np.nan if v is None else v for v in values], dtype=float)
    for nm, rows in (("e", [0]), ("e or tie", [0, 1])):
        wt = h[rows].sum(axis=0); ok = ~np.isnan(vals) & (wt > 0)
        if wt[ok].sum() <= 0:
            print(f"      e-distribution ({nm}): empty"); continue
        mean = float(vals[ok] @ wt[ok]/wt[ok].sum())
        qv = wquant(vals[ok], wt[ok], QS)
        print(f"      e-distribution over cases '{nm}': mean {mean:.4f};  quantiles 5/25/50/75/95 %: "
              + "  ".join(f"{v:g}" for v in qv))
        out[f'mean_{nm}'] = mean; out[f'q_{nm}'] = qv
    return out

def report(Hs, values, title):
    F, R = Hs['face'], Hs['far']
    W = F.sum(); Weq = F[2].sum()
    print(f"\n==== {title} ====")
    print(f"  total weight over steps k = {KMIN}..{K-1}: {W:.6f}  (= number of steps {K-KMIN} x mass);"
          f"  weight with r == x_k: {Weq:.3e};  weight where an edge index falls off the grid: {Hs['clip']:.3e}")
    Wn = F[:2, 1].sum(); Wf = F[:2, 0].sum()
    print(f"  item 4: fraction with |r - x_k| < sigma: {Wn/W:.4f};  with |r - x_k| >= sigma: {Wf/W:.4f}")
    res = {}
    print("  items 1-2: facing edge, all steps (fractions of the total weight)")
    res['face_all'] = edge_block(F, values, "facing edge", (slice(0, 2), slice(None)), W)
    print("  item 3: far edge, all steps (fractions of the total weight)")
    res['far_all'] = edge_block(R, values, "far edge", (slice(0, 2), slice(None)), W)
    print("  item 4: facing edge, restricted to |r - x_k| >= sigma (fractions of that weight)")
    res['face_farr'] = edge_block(F, values, "facing edge, |r-x_k| >= sigma", (slice(0, 2), 0), Wf)
    print("  (extra) facing edge, restricted to |r - x_k| < sigma (fractions of that weight)")
    res['face_near'] = edge_block(F, values, "facing edge, |r-x_k| < sigma", (slice(0, 2), 1), Wn)
    print("  (extra) far edge, restricted to |r - x_k| < sigma and to |r - x_k| >= sigma (fractions of that weight)")
    res['far_near'] = edge_block(R, values, "far edge, |r-x_k| < sigma", (slice(0, 2), 1), Wn)
    res['far_farr'] = edge_block(R, values, "far edge, |r-x_k| >= sigma", (slice(0, 2), 0), Wf)
    Jt = Hs['joint']
    for nm, sl, dn in (("all", slice(None), W), ("|r-x_k| < sigma", 1, Wn), ("|r-x_k| >= sigma", 0, Wf)):
        jj = Jt[:2, sl].reshape(-1, 3, 3).sum(axis=0)
        print(f"  (extra) both edges set by a candidate e (status 'e' on both sides), {nm}: {jj[0, 0]/dn:.4f}"
              f"   (fraction of that weight)")
    print("  (extra) facing edge by side: r < x_k (facing = upper) and r > x_k (facing = lower)")
    for s, nm in ((0, "r < x_k"), (1, "r > x_k")):
        Ws = F[s].sum()
        edge_block(F, values, f"facing edge, {nm}", (s, slice(None)), Ws)
    print("  mass-weighted fraction of cases with an edge set by a candidate e below 0.5 or negative (status 'e';"
          " fractions of the total weight):")
    for nm_, v_ in zip(LOW_NAMES, Hs['low']):
        print(f"    {nm_:32s} {v_/W:.6f}")
    # histogram of the facing e and far e
    hf_all = F[:2, :, 0].sum(axis=(0, 1)); hf_far = F[:2, 0, 0].sum(axis=0)
    hr_all = R[:2, :, 0].sum(axis=(0, 1))
    print("  weighted histogram of the e that sets the edge (status 'e' only); fractions of the total weight W,")
    print("  and for the facing edge also of the weight with |r - x_k| >= sigma")
    print("        e    facing/W   facing,|r-x|>=sig / W_far   far/W")
    for j, v in enumerate(values):
        if v is None: continue
        if hf_all[j] + hr_all[j] > 0:
            print(f"    {v:6g}   {hf_all[j]/W:8.5f}   {hf_far[j]/Wf:8.5f}                     {hr_all[j]/W:8.5f}")
    return res

# ---------------- recompute baseline and the three rounds ----------------
hw = cc.FAC*20
w0, B, Eq0, mass = cc.forward(lambda k: (np.clip(idx+1-hw, 0, n+1), np.clip(idx+1+hw, 0, n+1)))
ld0, J0 = cc.logdet(w0)
print(f"constant SoD: E[N] = {B:.4f}, log det J = {ld0:.4f}   (run: E[N] = {m0.group(1)}, log det J = {m0.group(2)};"
      f" equal to the printed digits: {f'{B:.4f}' == m0.group(1) and f'{ld0:.4f}' == m0.group(2)})"
      f"   ({time.time()-t0:.0f} s)", flush=True)

A = np.linalg.inv(J0)
for rnd, (ent, best_ld_s, best_EN_s) in enumerate(runs):
    gam = np.einsum('ki,ij,kj->k', cc.H, A, cc.H)/cc.sig**2
    gbar = gam[1:].mean()
    lo, hi = None, None; lam = 3.0*gbar; lams = []
    for (lg_s, EN_p, ld_p, V_s) in ent:
        if abs(EN_p - B) < 1e-3:
            sys.exit(f"round {rnd+1}: printed E[N] {EN_p} too close to B = {B} to read the feasibility decision")
        assert f"{lam/gbar:8.4f}".strip() == lg_s, (rnd, lam/gbar, lg_s)
        lams.append(lam)
        if EN_p <= B: hi = lam
        else: lo = lam
        if lo is None: lam = hi*0.7
        elif hi is None: lam = lo*1.5
        else: lam = 0.5*(lo + hi)
    feas = [j for j, e in enumerate(ent) if e[1] <= B]
    lds = sorted((ent[j][2] for j in feas), reverse=True)
    if len(lds) > 1 and lds[0] - lds[1] < 2e-4:
        sys.exit(f"round {rnd+1}: best feasible entry not unique at the printed precision")
    jb = max(feas, key=lambda j: ent[j][2])
    lg_s, EN_p, ld_p, V_s = ent[jb]
    lam = lams[jb]
    Vs, CL, CU = cc.backward(gam, lam)
    w, EN, Eq, mass = cc.forward(cc.table_win(CL, CU))
    ld = cc.logdet(w)[0]
    ok = (f"{EN:.3f}" == best_EN_s and f"{ld:.4f}" == best_ld_s and f"{Vs:.6f}" == V_s)
    print(f"round {rnd+1}: replayed {len(ent)} lambdas, best feasible = entry {jb+1}, lambda/gbar = {lam/gbar:.10f}"
          f" (printed {lg_s});  recomputed E[N] = {EN:.3f}, log det J = {ld:.4f}, V* = {Vs:.6f}"
          f"  (run: {best_EN_s}, {best_ld_s}, {V_s});  equal to the printed digits: {ok}   ({time.time()-t0:.0f} s)",
          flush=True)
    if rnd < 2:
        A = np.linalg.inv(cc.logdet(w)[1]); del CL, CU

# ---------------- forward passes with statistics ----------------
values_T = list(E) + [None]
HT = new_hist(nE, values_T)
wT, ENT, _, massT = forward_stats(table_win_stats(CL, CU), HT)
print(f"\nforward pass with statistics, round-3 table: E[N] = {ENT:.3f}, log det J = {cc.logdet(wT)[0]:.4f},"
      f" mass {massT:.12f};  w identical to forward(): {np.array_equal(wT, w)}   ({time.time()-t0:.0f} s)", flush=True)
resT = report(HT, values_T, f"round-3 table (FAC {cc.FAC}, {SET_s}, MIN = {MIN_s})")

e_rule = float(rule_e_s.group(1))
HR = new_hist(2, [e_rule, None])
wR, ENR, _, massR = forward_stats(rule_win_stats(e_rule), HR)
ldR = cc.logdet(wR)[0]
print(f"\nforward pass with statistics, one-parameter rule at e = {e_rule} (the printed, 4-decimal value):"
      f" E[N] = {ENR:.3f}, log det J = {ldR:.4f}  (run: e = {rule_e_s.group(1)}, {rule_e_s.group(2)},"
      f" E[N] = {rule_e_s.group(3)}; equal to the printed digits:"
      f" {f'{ldR:.4f}' == rule_e_s.group(2) and f'{ENR:.3f}' == rule_e_s.group(3)})   ({time.time()-t0:.0f} s)", flush=True)
resR = report(HR, [e_rule, None], f"one-parameter rule e = {e_rule} (FAC {cc.FAC}, {SET_s}, MIN = {MIN_s})")

if OUT:
    np.savez(OUT, E=np.array(E), face_T=HT['face'], far_T=HT['far'], face_R=HR['face'], far_R=HR['far'],
             e_rule=e_rule, MIN=float(MIN_s), FAC=cc.FAC, KMIN=KMIN, low_T=HT['low'], low_R=HR['low'],
             low_names=np.array(LOW_NAMES))
print(f"\ntime {time.time()-t0:.0f} s")
