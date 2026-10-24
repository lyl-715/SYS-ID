"""Round 4, Part A: general-window evaluator (spec reports_CN/plan_round4_CN.md v3.1, frozen; §1.2, §3 items 1, 2, 6).

A generalisation of the ACCEPTED reference-value-grid operator (density_recursion_v1.py version B on the window path of
evaluator_fast.py), not of the offset-grid MDPs of rounds 2a-2d. Grid r_i = z_0 + i h, h = sigma/24, range
[min x - 8 Delta, max x + 8 Delta] (dr.make_grid, unchanged); discrete kernel ghat_{k,j} = dr._kernel on the step's support
window, normalised there (unchanged); c from dr.cfun (unchanged) with alpha, beta of the node's own window; record, J assembly
and invariants from dr.evaluate (unchanged; its run_ab is replaced for one call only, as evaluator_fast.evaluate_fast does).

Window table: at step k and node i the no-send window is (r_i - M^-_{k,i} h, r_i + M^+_{k,i} h), integer half-widths in grid
units, omega_{k,ij} = 1 inside, 1/2 at j - i in {-M^-, M^+}, 0 outside (end-point half weights as in version B).

Two code paths (spec §3 item 1):
  - every window equals (480, 480) = (20 sigma, 20 sigma) at h = sigma/24: evaluator_fast.evaluate_fast, the accepted operator,
    called as is (support half-width R = Delta + 48 sigma = 68 sigma);
  - otherwise the general path below, support half-width R = Delta_max + rho + 5 sigma + 4 sigma = 40 + 39 + 5 + 4 = 88 sigma.

General step k on the support window [a_k, b_k] (spec §1.2; extended arrays over [a_k - X, b_k + X], X = largest half-width):
  ghat tails      L(j) = sum_{j' < j} ghat_j',  S(j) = sum_{j' > j} ghat_j'  (one prefix pass, one suffix pass, non-negative terms)
  stay            Phat_i = [L(u_i) - L(l_i + 1)] + 1/2 (ghat_{l_i} + ghat_{u_i}),     l_i = i - M^-_{k,i}, u_i = i + M^+_{k,i}
  send            qhat_i = [L(l_i) + S(u_i)] + 1/2 (ghat_{l_i} + ghat_{u_i})            (left tail + right tail; never 1 - Phat)
  arrival         T_j = sum_{l_i > j} m_i + 1/2 sum_{l_i = j} m_i + sum_{u_i < j} m_i + 1/2 sum_{u_i = j} m_i
                  (m_i bucketed by l_i and by u_i, one suffix sum and one prefix sum; no "total minus inside")
  update          m_{k+1,j} = m_{k,j} Phat_{k,j} + ghat_{k,j} T_j,   m_{2,j} = ghat_{1,j};  E[N] = 1 + sum_k sum_i m_{k,i} qhat_{k,i}
  information     w_k = sum_i m_{k,i} c(alpha_{k,i}, beta_{k,i}),  alpha = (r_i - M^- h - x_k)/sigma,  beta = (r_i + M^+ h - x_k)/sigma
Assertions kept from the accepted evaluator: |sum m - 1| <= 1e-10 and min m >= 0 at every step, the support-edge assertion
(no non-zero mass within a band of X nodes outside the next support window), the initial support inside W_2.

Backward fixed-policy evaluation (for the explicit-matrix check and for check B4 of trigger_dp.py) uses the same tails:
  V_k(i) = rho_{k,i} + Phat_{k,i} V_{k+1}(i) + [Lv(l_i) + Sv(u_i)] + 1/2 (gv_{l_i} + gv_{u_i}),  gv = ghat V_{k+1},  V_{K+1} = 0;
V_{k+1} is taken as 0 outside the support window W_{k+1} (nodes with zero mass under every table; DECISION A-3).

Usage (from this directory):  python3 evaluator_window.py --check    # Part A checks -> logs/partA_checks.log, results/partA_checks.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
sys.path.insert(0, EV)
import numpy as np
from numba import njit
import density_recursion_v1 as dr
import evaluator_fast as ef

LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)
s, D, K = dr.s, dr.D, dr.K
N_GRID = 24                                      # accepted grid h = sigma/24
SOD_M = 20*N_GRID                                # 480 grid units = 20 sigma = Delta
STEP_M = 2*N_GRID                                # 48 grid units = 2 sigma (action spacing)
N_ACT1 = 20                                      # M^+- in {48, 96, ..., 960}
X24 = N_ACT1*STEP_M                              # 960 = 40 sigma = Delta_max
R_SUP = 88*s                                     # spec §1.2: Delta_max + rho + 5 sigma + 4 sigma
B_BUDGET = 316.8921168493721                     # round2a/config_recheck.json budget_B (archived full precision)

# P5514 from the round 2a archive, in the evaluator's in-memory input table only (as round2a/recheck7.py does)
CFG2A = json.load(open(os.path.join(EV, 'round2a', 'config_recheck.json')))
_P = CFG2A['candidates'][0]; assert _P['id'] == 'NM-none-P5514'
dr.CFG['inputs']['P5514'] = dict(a=_P['a'], chi=_P['chi'])
assert CFG2A['budget_B']['value'] == B_BUDGET
INPUTS = ('U', 'P5514')


class AssertionStop(RuntimeError):
    """an assertion of the accepted evaluator failed on the general path: the run stops (no fallback; DECISION A-4)."""


def windows88(xk, z0, h, G, R=R_SUP):
    """support windows [a_k, b_k] as evaluator_fast.windows, with R = 88 sigma: a_k = ceil((x_k - R - z_0)/h),
    b_k = floor((x_k + R - z_0)/h), clipped to [0, G-1]."""
    a = np.ceil((xk-R-z0)/h).astype(np.int64); b = np.floor((xk+R-z0)/h).astype(np.int64)
    return np.clip(a, 0, G-1), np.clip(b, 0, G-1)


# --- numba kernels (accepted flags: @njit(cache=True), no fastmath, no parallel) ---------------------------------------
@njit(cache=True)
def _tails(v, a, b, X, vE, LE, SE):
    """extended arrays over [a - X, b + X] (index e = j - (a - X)) of v restricted to [a, b] (zero elsewhere, also outside
    the grid): vE[e] = v_j, LE[e] = sum_{e' < e} vE[e'] (prefix pass), SE[e] = sum_{e' > e} vE[e'] (suffix pass)."""
    ng = v.size; L = b-a+1+2*X; base = a-X
    for e in range(L):
        j = base+e
        vE[e] = v[j] if (j >= a and j <= b and j >= 0 and j < ng) else 0.0
    LE[0] = 0.0
    for e in range(1, L): LE[e] = LE[e-1]+vE[e-1]
    SE[L-1] = 0.0
    for e in range(L-2, -1, -1): SE[e] = SE[e+1]+vE[e+1]

@njit(cache=True)
def _gen_step(m, mn, gh, Mm, Mp, a, b, ap, bp, an, bn, X, gE, LgE, SgE, Bl, Bu, SBl, PBu, Ph):
    """one general step on [a, b]; m = source (m_k, not modified), mn = destination. Destination zeroed on [ap, bp] (the
    previous window, as evaluator_fast._ab_step_w), written on [a, b]. Returns (sum_i m_i qhat_i, sum mn, min mn, first and
    last j with mn_j != 0, number of non-zero mn_j, support-edge assertion ok: mn == 0 within X nodes outside [an, bn])."""
    ng = m.size; base = a-X; nw = b-a+1
    _tails(gh, a, b, X, gE, LgE, SgE)
    for j in range(ap, bp+1): mn[j] = 0.0
    nb = nw+2                                            # bucket index t = j - (a - 1), j in [a - 1, b + 1]
    for t in range(nb): Bl[t] = 0.0; Bu[t] = 0.0
    en = 0.0
    for o in range(nw):
        i = a+o; l = i-Mm[o]; u = i+Mp[o]
        el = l-base; eu = u-base
        half = 0.5*(gE[el]+gE[eu])
        q = (LgE[el]+SgE[eu])+half
        Ph[o] = (LgE[eu]-LgE[el+1])+half
        en += m[i]*q
        lc = min(max(l, a-1), b+1); uc = min(max(u, a-1), b+1)   # l < a behaves as a - 1, u > b as b + 1 for every j in [a, b]
        Bl[lc-(a-1)] += m[i]; Bu[uc-(a-1)] += m[i]
    SBl[nb] = 0.0
    for t in range(nb-1, -1, -1): SBl[t] = SBl[t+1]+Bl[t]  # SBl[t] = sum_{t' >= t} Bl
    PBu[0] = 0.0
    for t in range(1, nb): PBu[t] = PBu[t-1]+Bu[t-1]        # PBu[t] = sum_{t' < t} Bu
    for o in range(nw):
        j = a+o; t = o+1
        T = (SBl[t+1]+PBu[t])+0.5*(Bl[t]+Bu[t])
        mn[j] = m[j]*Ph[o]+gh[j]*T
    tot = 0.0; mmin = mn[a]; slo = -1; shi = -1; nnz = 0
    for j in range(a, b+1):
        v = mn[j]; tot += v
        if v < mmin: mmin = v
        if v != 0.0:
            nnz += 1
            if slo < 0: slo = j
            shi = j
    ok = True
    for j in range(max(an-X, 0), an):
        if mn[j] != 0.0: ok = False
    for j in range(bn+1, min(bn+X, ng-1)+1):
        if mn[j] != 0.0: ok = False
    return en, tot, mmin, slo, shi, nnz, ok

@njit(cache=True)
def _gen_back(Vn, an, bn, gh, Mm, Mp, a, b, X, rc, wq, gE, LgE, SgE, gvE, LgvE, SgvE, VnE, Vout):
    """backward fixed-policy step on [a, b]: Vout[o] = rc[o] + wq qhat + Phat V_{k+1}(i) + send-branch sum, V_{k+1} = Vn on
    [an, bn] (0 outside). The send-branch sum = left tail + right tail of gv = ghat V_{k+1} (prefix and suffix passes)."""
    ng = gh.size; base = a-X; nw = b-a+1; L = nw+2*X
    _tails(gh, a, b, X, gE, LgE, SgE)
    for e in range(L):
        j = base+e
        VnE[e] = Vn[j-an] if (j >= an and j <= bn) else 0.0
        gvE[e] = gE[e]*VnE[e]
    LgvE[0] = 0.0
    for e in range(1, L): LgvE[e] = LgvE[e-1]+gvE[e-1]
    SgvE[L-1] = 0.0
    for e in range(L-2, -1, -1): SgvE[e] = SgvE[e+1]+gvE[e+1]
    for o in range(nw):
        i = a+o; el = i-Mm[o]-base; eu = i+Mp[o]-base; ei = o+X
        half = 0.5*(gE[el]+gE[eu])
        q = (LgE[el]+SgE[eu])+half
        P = (LgE[eu]-LgE[el+1])+half
        snd = (LgvE[el]+SgvE[eu])+0.5*(gvE[el]+gvE[eu])
        Vout[o] = rc[o]+wq*q+P*VnE[ei]+snd


def _warmup():
    ef._warmup()
    ng = 64; m = np.zeros(ng); m[20:40] = 0.05; mn = np.zeros(ng); gh = np.zeros(ng); gh[10:50] = 1/40
    X = 4; nw = ng; L = nw+2*X
    Mm = np.full(nw, 3, np.int64); Mp = np.full(nw, 2, np.int64)
    bufs = [np.empty(L) for _ in range(3)]; bk = [np.empty(nw+3) for _ in range(4)]; Ph = np.empty(nw)
    _gen_step(m, mn, gh, Mm, Mp, 0, ng-1, 0, ng-1, 0, ng-1, X, *bufs, *bk, Ph)
    _gen_back(np.zeros(nw), 0, ng-1, gh, Mm, Mp, 0, ng-1, X, np.zeros(nw), 1.0, *[np.empty(L) for _ in range(7)], np.empty(nw))


# --- window tables -------------------------------------------------------------------------------------------------
# A table is an object with .rows(p, a, b) -> (M^-, M^+) int64 arrays for the nodes a..b of array index p (step k = p + 1,
# p = 1..K-1), .X (largest half-width, grid units) and .is_sod() (every window equals (480, 480) at h = sigma/24).
class ConstTable:
    def __init__(self, mm, mp): self.mm, self.mp = int(mm), int(mp); self.X = max(self.mm, self.mp)
    def rows(self, p, a, b): n = b-a+1; return np.full(n, self.mm, np.int64), np.full(n, self.mp, np.int64)
    def is_sod(self): return self.mm == SOD_M and self.mp == SOD_M

class StepTable:
    """k-dependent windows (constant over nodes): mm[p], mp[p] for array index p."""
    def __init__(self, mm, mp): self.mm, self.mp = np.asarray(mm, np.int64), np.asarray(mp, np.int64); self.X = int(max(self.mm.max(), self.mp.max()))
    def rows(self, p, a, b): n = b-a+1; return np.full(n, self.mm[p], np.int64), np.full(n, self.mp[p], np.int64)
    def is_sod(self): return False

class FullTable:
    """(k, i)-dependent windows on the whole grid: mm[p, i], mp[p, i] (small-grid check)."""
    def __init__(self, mm, mp): self.mm, self.mp = np.asarray(mm, np.int64), np.asarray(mp, np.int64); self.X = int(max(self.mm.max(), self.mp.max()))
    def rows(self, p, a, b): return self.mm[p, a:b+1].copy(), self.mp[p, a:b+1].copy()
    def is_sod(self): return False

def code_to_m(code):
    """action code c = i_- * 20 + i_+ (i_+- = 0..19) -> (M^-, M^+) = (48 (i_- + 1), 48 (i_+ + 1)) grid units (DECISION B-2)."""
    c = np.asarray(code, np.int64); return STEP_M*(c//N_ACT1+1), STEP_M*(c % N_ACT1+1)

SOD_CODE = (SOD_M//STEP_M-1)*N_ACT1+(SOD_M//STEP_M-1)          # (480, 480) -> 9*20 + 9 = 189

class CodeTable:
    """a DP table: codes[r, o] uint16 for step k = r + 2 (array index p = r + 1) at node start[r] + o (o < width[r])."""
    def __init__(self, codes, start, width):
        self.codes, self.start, self.width = codes, np.asarray(start, np.int64), np.asarray(width, np.int64); self.X = X24
    def rows(self, p, a, b):
        r = p-1; assert a == self.start[r] and b-a+1 == self.width[r], (p, a, b, self.start[r], self.width[r])
        return code_to_m(self.codes[r, :self.width[r]])
    def is_sod(self): return all(bool(np.all(self.codes[r, :self.width[r]] == SOD_CODE)) for r in range(len(self.start)))

class FineTable:
    """spec §1.4 refinement transport of a coarse (h = sigma/24) table to the sigma/48 grid: fine node j looks up the nearest
    coarse node, ties -> smaller index (exact grid relation z^f_{2i} = z^c_i, so nearest = j // 2), outside the stored range
    -> the boundary node of that step's stored range; the PHYSICAL half-widths are kept (grid counts doubled) and the window
    is centred at the fine node's own r."""
    def __init__(self, coarse):
        self.c = coarse; self.X = 2*coarse.X
    def rows(self, p, a, b):
        r = p-1; s0 = self.c.start[r]; w = self.c.width[r]
        ic = np.clip(np.arange(a, b+1)//2, s0, s0+w-1)-s0
        mm, mp = code_to_m(self.c.codes[r, ic])
        return 2*mm, 2*mp
    def is_sod(self): return False


# --- general forward and backward ----------------------------------------------------------------------------------
def _buffers(nw_max, X):
    L = nw_max+2*X
    return [np.empty(L) for _ in range(3)], [np.empty(nw_max+3) for _ in range(4)], np.empty(nw_max)

def forward(z, dz, sg, xk, a, b, table, gam=None, record=False, record_window=False):
    """general forward on nodes z (z_i = z_0 + i dz), measurement sd sg, outputs xk (len Kn), support windows [a_p, b_p].
    Returns E[N] (discrete qhat), E[N] with exact P (diagnostic, as the accepted record), w_k, E[q_k], per-step mass, min m,
    kernel mass, support, assertion flags; with record=True also every m_k (k = 2..Kn+1) on the full grid; with
    record_window=True m_k on [a_p, b_p] as float32 for p = 1..Kn-1 (plots only; the arithmetic is unchanged)."""
    ng = z.size; Kn = xk.size; X = table.X
    nw_max = int((b-a+1).max()); gbuf, bbuf, Ph = _buffers(nw_max, X)
    m = np.zeros(ng); mn = np.zeros(ng); gh = np.zeros(ng)
    EN = 1.0; ENP = 1.0; W = np.zeros(Kn); W[0] = 1.0; Q = np.zeros(Kn); Q[0] = 1.0
    mass = np.empty(Kn); minm = np.empty(Kn); kmass = np.empty(Kn)
    nnz = np.zeros(Kn, np.int64); slo = np.zeros(Kn, np.int64); shi = np.zeros(Kn, np.int64); ok = np.ones(Kn, np.bool_)
    ms = [] if record else None; mw = [None]
    g1 = np.empty(ng); kmass[0] = dr._kernel(z, xk[0], sg, dz, g1)
    nz = np.flatnonzero(g1)
    if not (nz[0] >= a[1] and nz[-1] <= b[1]):
        raise AssertionStop(f"initialisation: support of m_2 [{nz[0]}, {nz[-1]}] not inside W_2 [{a[1]}, {b[1]}]")
    m[a[1]:b[1]+1] = g1[a[1]:b[1]+1]
    nz = np.flatnonzero(m); lo, hi = nz[0], nz[-1]
    mass[0] = m.sum(); minm[0] = m.min(); nnz[0] = nz.size; slo[0], shi[0] = lo, hi
    if record: ms.append(m.copy())
    for p in range(1, Kn):
        A, Bb = int(a[p]), int(b[p])
        if lo < A or hi > Bb:
            raise AssertionStop(f"step {p+1}: support of m_k [{lo}, {hi}] not inside the support window [{A}, {Bb}]")
        Mm, Mp = table.rows(p, A, Bb)
        if record_window: mw.append(m[A:Bb+1].astype(np.float32))
        zz = z[lo:hi+1]; mm = m[lo:hi+1]; o0, o1 = lo-A, hi-A+1
        al = (zz-Mm[o0:o1]*dz-xk[p])/sg; be = (zz+Mp[o0:o1]*dz-xk[p])/sg
        cev, cno = dr.cfun(al, be)
        W[p] = mm@(cev+cno)
        P = np.where(al > 0, dr.ndtr(-al)-dr.ndtr(-be), dr.ndtr(be)-dr.ndtr(al))
        ENP += mm@(1-P)
        gh[a[p-1]:b[p-1]+1] = 0.0
        kmass[p] = dr._kernel(z[A:Bb+1], xk[p], sg, dz, gh[A:Bb+1])
        an, bn = (int(a[p+1]), int(b[p+1])) if p < Kn-1 else (A, Bb)
        en, mass[p], minm[p], lo, hi, nnz[p], ok[p] = _gen_step(m, mn, gh, Mm, Mp, A, Bb, int(a[p-1]), int(b[p-1]), an, bn, X,
                                                               *gbuf, *bbuf, Ph)
        if not ok[p]:
            raise AssertionStop(f"support-edge assertion failed at step {p+1}: non-zero mass within {X} nodes outside [{an}, {bn}]")
        if abs(mass[p]-1) > 1e-10 or minm[p] < 0:
            raise AssertionStop(f"step {p+1}: |sum m - 1| = {abs(mass[p]-1)!r} or min m = {minm[p]!r} violates the accepted assertion")
        slo[p], shi[p] = lo, hi
        Q[p] = en; EN += en; m, mn = mn, m
        if record: ms.append(m.copy())
    out = dict(EN=EN, ENP=ENP, W=W, Q=Q, mass=mass, minm=minm, kmass=kmass, G=ng, M='window table', Delta_over_h=None,
               win_a=a, win_b=b, nnz=nnz, sup_lo=slo, sup_hi=shi, assert_ok=ok)
    if gam is not None: out['Fgam'] = float(np.asarray(gam)[1:]@W[1:])
    if record: out['m_steps'] = ms
    if record_window: out['m_window'] = mw
    return out

def backward(z, dz, sg, xk, a, b, table, gam, record=False):
    """backward fixed-policy evaluation of sum_{k>=2} gam_k w_k and of N = sum_{k>=2} E[q_k] separately (V^c, V^q), each
    started from m_2 = ghat_1 on W_2: returns (sum_i ghat_1(i) V^c_2(i), sum_i ghat_1(i) V^q_2(i)), optionally all V_k."""
    ng = z.size; Kn = xk.size; X = table.X
    nw_max = int((b-a+1).max()); L = nw_max+2*X
    ext = [np.empty(L) for _ in range(7)]
    gh = np.zeros(ng); Vc = np.zeros(1); Vq = np.zeros(1); an, bn = 0, -1
    Vcs, Vqs = ([], []) if record else (None, None)
    for p in range(Kn-1, 0, -1):
        A, Bb = int(a[p]), int(b[p]); nw = Bb-A+1
        Mm, Mp = table.rows(p, A, Bb)
        zz = z[A:Bb+1]; al = (zz-Mm*dz-xk[p])/sg; be = (zz+Mp*dz-xk[p])/sg
        cev, cno = dr.cfun(al, be); rc = gam[p]*(cev+cno)
        gh[:] = 0.0; dr._kernel(z[A:Bb+1], xk[p], sg, dz, gh[A:Bb+1])
        Vc_new = np.empty(nw); Vq_new = np.empty(nw)
        _gen_back(Vc, an, bn, gh, Mm, Mp, A, Bb, X, rc, 0.0, *ext, Vc_new)
        _gen_back(Vq, an, bn, gh, Mm, Mp, A, Bb, X, np.zeros(nw), 1.0, *ext, Vq_new)
        Vc, Vq, an, bn = Vc_new, Vq_new, A, Bb
        if record: Vcs.append((p, A, Vc.copy())); Vqs.append((p, A, Vq.copy()))
    g1 = np.empty(ng); dr._kernel(z, xk[0], sg, dz, g1)
    g2 = g1[a[1]:b[1]+1]
    out = dict(Fc=float(g2@Vc), Nq=float(g2@Vq))
    if record: out['Vc'] = Vcs[::-1]; out['Vq'] = Vqs[::-1]
    return out


# --- design-grid evaluation (record and J from the accepted dr.evaluate) --------------------------------------------
def design_grid(name, n=N_GRID, margin_mult=8):
    xk, Hk = dr.input_arrays(name); z, dz = dr.make_grid(xk, n, margin_mult*D, 'B'); ng = len(z)
    a, b = windows88(xk, z[0], dz, ng)
    return xk, Hk, z, dz, a, b

def run_general(xk, n, margin, table):
    z, dz = dr.make_grid(xk, n, margin, 'B'); ng = len(z)
    a, b = windows88(xk, z[0], dz, ng)
    assert int((a-table.X).min()) >= 0 and int((b+table.X).max()) <= ng-1, "a window would cross the design grid"
    return forward(z, dz, s, xk, a, b, table)

def evaluate_table(name, table, n=N_GRID, force_general=False):
    """spec §3 item 1: every window (480, 480) at n = 24 -> evaluator_fast.evaluate_fast unchanged (accepted operator);
    otherwise the general path inside the accepted dr.evaluate (record, J, invariants, timing)."""
    if n == N_GRID and table.is_sod() and not force_general:
        rec, J, r = ef.evaluate_fast(name, n); rec['path'] = 'accepted operator (evaluator_fast.evaluate_fast)'
        return rec, J, r
    full = dr.run_ab
    dr.run_ab = lambda xk, n_, margin, version: run_general(xk, n_, margin, table)
    try: rec, J, r = dr.evaluate('B', name, n, 8)
    finally: dr.run_ab = full
    rec['path'] = 'general window path (R = 88 sigma)'
    return rec, J, r

def rel_record(rec, J, J_same_sod, J_C):
    """Delta m_D against the same input's constant-Delta SoD (same grid); nine lambda_i and m_E against J_C (uniform input,
    constant-Delta SoD)."""
    lam = dr.eigh(J, J_C, eigvals_only=True)
    rec.update(dmD=float(rec['logdet']-np.linalg.slogdet(J_same_sod)[1]), lam=[float(v) for v in lam], mE=float(lam[0]))
    return rec


# --- Part A checks (spec §3 items 1 and 2) -------------------------------------------------------------------------
def _relerr(y, yref):
    y = np.asarray(y, float); yref = np.asarray(yref, float)
    return float(np.max(np.abs(y-yref))/max(float(np.max(np.abs(yref))), 1e-300))

SG_G, SG_K, SG_SEED = 200, 30, 20261024

def small_grid_setup():
    """spec §3 item 2: G = 200, h = sigma/24, z_0 = -4 sigma, K = 30, x_k = 2 sigma sin(2 pi k/30), k = 1..30 (sigma = the
    scenario sigma, DECISION A-5); tables (a), (b), (c) and the random gamma (DECISION A-6)."""
    h = s/24; z = -4*s+np.arange(SG_G)*h
    kk = np.arange(1, SG_K+1); xk = 2*s*np.sin(2*np.pi*kk/30)
    p = np.arange(SG_K); kstep = p+1                                   # array index p <-> step k = p + 1
    ta = ConstTable(24, 24)
    mmb = 12+6*(kstep % 4); mpb = 48-6*(kstep % 5); tb = StepTable(mmb, mpb)
    rng = np.random.default_rng(SG_SEED)
    mmc = np.zeros((SG_K, SG_G), np.int64); mpc = np.zeros((SG_K, SG_G), np.int64)
    mmc[1:] = rng.integers(6, 61, size=(SG_K-1, SG_G)); mpc[1:] = rng.integers(6, 61, size=(SG_K-1, SG_G))   # k = 2..30 rows
    tc = FullTable(mmc, mpc)
    gam = np.ones(SG_K); gam[1:] = rng.uniform(0.5, 2.0, size=SG_K-1)
    a = np.zeros(SG_K, np.int64); b = np.full(SG_K, SG_G-1, np.int64)
    return z, h, xk, a, b, dict(a=ta, b=tb, c=tc), gam

def dense_eval(z, h, sg, xk, table, gam):
    """explicit transition matrices: T_k[i, j] = delta_ij Phat_i + ghat_j (1 - omega_ij), Phat_i = sum_j ghat_j omega_ij and
    qhat_i = sum_j ghat_j (1 - omega_ij) as dense sums; forward m_{k+1} = m_k T_k, backward V_k = rho_k + T_k V_{k+1}."""
    G = z.size; Kn = xk.size; idx = np.arange(G)
    g1 = np.empty(G); dr._kernel(z, xk[0], sg, h, g1)
    m = g1.copy(); ms = [m.copy()]; EN = 1.0; F = 0.0; Ts = []; qs = []; cs = []
    for p in range(1, Kn):
        Mm, Mp = table.rows(p, 0, G-1)
        g = np.empty(G); dr._kernel(z, xk[p], sg, h, g)
        d = idx[None, :]-idx[:, None]
        om = np.where((d > -Mm[:, None]) & (d < Mp[:, None]), 1.0, 0.0)+np.where((d == -Mm[:, None]) | (d == Mp[:, None]), 0.5, 0.0)
        Ph = om@g; qh = (1.0-om)@g
        T = np.diag(Ph)+g[None, :]*(1.0-om)
        al = (z-Mm*h-xk[p])/sg; be = (z+Mp*h-xk[p])/sg; cev, cno = dr.cfun(al, be); c = cev+cno
        EN += m@qh; F += gam[p]*(m@c)
        m = m@T; ms.append(m.copy()); Ts.append(T); qs.append(qh); cs.append(c)
    Vc = np.zeros(G); Vq = np.zeros(G); Vcs = []; Vqs = []
    for p in range(Kn-1, 0, -1):
        T = Ts[p-1]; Vc = gam[p]*cs[p-1]+T@Vc; Vq = qs[p-1]+T@Vq; Vcs.append(Vc.copy()); Vqs.append(Vq.copy())
    return dict(m_steps=ms, EN=EN, F=F, Vc=Vcs[::-1], Vq=Vqs[::-1], ENb=1.0+g1@Vq, Fb=g1@Vc)

def small_grid_check(out):
    z, h, xk, a, b, tables, gam_r = small_grid_setup()
    rows = []; ok_all = True
    for tn, tab in tables.items():
        ft = forward(z, h, s, xk, a, b, tab, record=True)
        for gname, gam in (('gamma=1', np.ones(SG_K)), ('gamma=rand', gam_r)):
            dn = dense_eval(z, h, s, xk, tab, gam)
            bt = backward(z, h, s, xk, a, b, tab, gam, record=True)
            Ff = float(gam[1:]@ft['W'][1:])
            e_m = max(_relerr(x, y) for x, y in zip(ft['m_steps'], dn['m_steps']))
            e_Vc = max(_relerr(v[2], y) for v, y in zip(bt['Vc'], dn['Vc']))
            e_Vq = max(_relerr(v[2], y) for v, y in zip(bt['Vq'], dn['Vq']))
            res = dict(table=tn, gamma=gname, err_m_steps=e_m, err_EN_fwd=_relerr(ft['EN'], dn['EN']), err_F_fwd=_relerr(Ff, dn['F']),
                       err_Vc_steps=e_Vc, err_Vq_steps=e_Vq, err_EN_bwd=_relerr(1.0+bt['Nq'], dn['ENb']), err_F_bwd=_relerr(bt['Fc'], dn['Fb']),
                       EN=ft['EN'], F=Ff, mass_dev_max=float(np.abs(ft['mass']-1).max()), min_m=float(ft['minm'].min()),
                       send_prob_min=float(ft['Q'][1:].min()))
            res['pass'] = all(res[k] <= 1e-12 for k in res if k.startswith('err_'))
            ok_all &= res['pass']; rows.append(res)
            for k, v in res.items():
                if k not in ('table', 'gamma'): out(f"A4-2 table={tn} {gname} {k} {v!r}")
    crossing = {}
    for tn, t in tables.items():
        ii = np.arange(SG_G); cnt = 0
        for p in range(1, SG_K):
            Mm, Mp = t.rows(p, 0, SG_G-1); cnt += int(((ii-Mm) < 0).sum()+((ii+Mp) > SG_G-1).sum())
        crossing[tn] = cnt
    for tn, v in crossing.items(): out(f"A4-2 table={tn} windows_crossing_grid_boundary (node-steps) {v}")
    return ok_all, rows, crossing

def regression_check(out):
    """spec §3 item 1: (i) dispatch to the accepted operator, bitwise; (ii) general path at (480, 480), frozen tolerances."""
    res = {}; ok_all = True
    sod = ConstTable(SOD_M, SOD_M)
    for name in INPUTS:
        t0 = time.perf_counter(); rec0, J0, r0 = ef.evaluate_fast(name, N_GRID); t_ref = time.perf_counter()-t0
        rec1, J1, r1 = evaluate_table(name, sod)
        d1 = dict(EN=rec1['EN']-rec0['EN'], S=rec1['S']-rec0['S'], J_maxabs=float(np.max(np.abs(J1-J0))), path=rec1['path'])
        p1 = d1['EN'] == 0 and d1['S'] == 0 and d1['J_maxabs'] == 0
        t0 = time.perf_counter(); rec2, J2, r2 = evaluate_table(name, sod, force_general=True); t_gen = time.perf_counter()-t0
        Jm = float(np.max(np.abs(J0)))
        jr = float(np.max(np.abs(J2-J0)/(np.abs(J0)+Jm)))
        d2 = dict(EN_rel=abs(rec2['EN']-rec0['EN'])/abs(rec0['EN']), S_rel=abs(rec2['S']-rec0['S'])/abs(rec0['S']), J_ratio_max=jr,
                  logdet_diff=rec2['logdet']-rec0['logdet'], EN_P_rel=float(abs(rec2['EN_P']-rec0['EN_P'])/abs(rec0['EN_P'])),
                  w_maxabs=float(np.max(np.abs(r2['W']-r0['W']))), path=rec2['path'],
                  mass_dev_max=rec2['mass_dev_max'], min_m=rec2['min_m'], assert_fail=int((~r2['assert_ok']).sum()),
                  runtime_ref_s=t_ref, runtime_general_s=t_gen)
        p2 = d2['EN_rel'] <= 1e-12 and d2['S_rel'] <= 1e-12 and jr <= 1e-12
        res[name] = dict(bitwise=d1, bitwise_pass=p1, general=d2, general_pass=p2, EN=rec0['EN'], S=rec0['S'], logdet=rec0['logdet'])
        ok_all &= p1 and p2
        out(f"A4-1 {name} reference evaluator_fast EN {rec0['EN']!r} S {rec0['S']!r} logdet {rec0['logdet']!r}")
        for k, v in d1.items(): out(f"A4-1 {name} (i) dispatch {k} {v!r}")
        out(f"A4-1 {name} (i) bitwise (difference 0 for E[N], S, 81 J elements) {'PASS' if p1 else 'FAIL'}")
        for k, v in d2.items(): out(f"A4-1 {name} (ii) general(480,480) {k} {v!r}")
        out(f"A4-1 {name} (ii) tolerance (E[N], S rel <= 1e-12; |dJ| <= 1e-12 (|ref| + max|J|)) {'PASS' if p2 else 'FAIL'}")
    return ok_all, res


def checks():
    sys.stdout = dr.Tee(os.path.join(LOGS, 'partA_checks.log'))
    import numba, scipy
    print(f"=== evaluator_window.py --check (round 4 Part A); spec reports_CN/plan_round4_CN.md v3.1; python {sys.version.split()[0]}, "
          f"numpy {np.__version__}, scipy {scipy.__version__}, numba {numba.__version__}; B = {B_BUDGET!r}")
    same_d = (dr.CFG['inputs']['P5514']['a'] == dr.CFG['inputs']['D']['a'] and dr.CFG['inputs']['P5514']['chi'] == dr.CFG['inputs']['D']['chi'])
    print(f"A0 P5514 a, chi from round2a/config_recheck.json candidates[0] (NM-none-P5514); identical to evaluator config.json input D: {same_d}")
    print(f"A0 support half-width R = 88 sigma = {R_SUP!r}; X = {X24} grid units = 40 sigma; SoD window (480, 480); action code of (480, 480) = {SOD_CODE}")
    for name in INPUTS:
        xk, Hk, z, dz, a, b = design_grid(name)
        print(f"A0 {name} G = {len(z)}, support window nodes min {int((b-a+1)[1:].min())} max {int((b-a+1)[1:].max())}, "
              f"min a_k - X {int((a-X24).min())}, max b_k + X {int((b+X24).max())} (G - 1 = {len(z)-1}); max |x_(k+1) - x_k| = {float(np.abs(np.diff(xk)).max()/s)!r} sigma")
    t0 = time.perf_counter(); _warmup(); print(f"A0 warm-up (JIT) {time.perf_counter()-t0:.1f} s")
    ok1, res1 = regression_check(print)
    t0 = time.perf_counter(); ok2, res2, cross = small_grid_check(print); t2 = time.perf_counter()-t0
    print(f"A4-2 small-grid explicit-matrix check (3 tables x 2 gamma, all errors <= 1e-12): {'PASS' if ok2 else 'FAIL'}; {t2:.1f} s")
    print(f"A4 check 1 (SoD regression, both paths): {'PASS' if ok1 else 'FAIL'}; check 2 (explicit matrices): {'PASS' if ok2 else 'FAIL'}")
    json.dump(dict(regression=res1, regression_pass=ok1, small_grid=res2, small_grid_pass=ok2, crossing=cross),
              open(os.path.join(RES, 'partA_checks.json'), 'w'), indent=1)
    return ok1 and ok2


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.check: sys.exit(0 if checks() else 1)
