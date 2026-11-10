"""[Round 13 copy of round11/eval11.py (prompt reports_CN/prompt_round13_part0_infrastructure_EN.md §2.2). Changes, and nothing
else: (1) in-memory registration of the inputs of round13/inputs13.json (config.json is not edited) with the check
|sum a_m^2 - 1| <= 1e-12; (2) NAME from the environment variable R13_INPUT (default 'U'); (3) fine_grid passes this input's
own coarse origin and step to round 7's cell_index; (4) two input-specific assertions after the module-level grid;
(5) DECISION 13-30 (user decision 2026-11-10, Part A): the sigma/24 conservation tolerance TOL24 = 1e-11 instead of 1e-12 for every
input. Mass is conserved identically in exact arithmetic; K eps = 5000 x 2.2e-16 = 1.1e-12 is the rounding floor of the recursion, so
1e-12 sat on that floor; 1e-11 is about 10 K eps. TOL120 stays 1e-10; non-negativity and zero mass at the grid ends stay strict.
The rest of this docstring is round 11's, unchanged.]

Round 11: stand-alone forward / backward evaluator on the FULL design grid (spec reports_CN/plan_round11_edge_actions_EN.md,
frozen 2026-11-09, §2.1 and DECISION 11-3; execution prompt reports_CN/prompt_round11_claude_code_EN.md §2.1).

The transition and score formulas are copied from the frozen round-4 general path (round4/evaluator_window.py: _tails,
_gen_step, _gen_back, forward, backward), which stays read-only and is imported only for its constants, its grid and the
accepted kernel / cfun of density_recursion_v1 (dr._kernel, dr.cfun, dr.ndtr). Unchanged: grid step h = sigma/24, the
measurement kernel (dr._kernel, normalised on the step's 88-sigma window ew.windows88, exactly zero beyond about 38.6 sigma),
the two-branch mass update (10)
    m_{k+1,j} = m_{k,j} Phat_{k,j} + ghat_{k,j} T_j,
the trapezoidal end-point weights (1/2 at j - i in {-L, U}), the "two tails, never 1 - Phat" rule (qhat = left tail + right
tail of ghat), the prefix/suffix-sum arrival term T_j, c from dr.cfun on the nodes that carry mass, J = sigma^-2 sum w_k h_k h_k^T.

Changed (DECISION 11-3, fixed by review):
  * reference-state support = the whole design grid [0, G-1] at every step (forward and backward); the measurement-kernel
    support is handled separately (the frozen 88-sigma window of the step, asserted to end in exact zeros);
  * window edges may lie anywhere in the padded index range [-X, G-1+X] (X = 960 grid units >= every edge overshoot of both
    action sets; asserted), i.e. the boundary index range is the full grid width, no 40-sigma cap;
  * assertions at every step: |sum m - 1| <= tol (1e-12 on sigma/24; 1e-10 on sigma/120, DECISION 11-5), min m >= 0, and
    zero mass at the first and last grid node. Any failure raises AssertionStop (no fallback).

Tables (prompt §2.2): action code per (k, i) on the full sigma/24 grid (uint16), with the action set stored alongside. An action
set is a product of a lower-edge list and an upper-edge list, code = i_lo * N_hi + i_hi (fixed tie order = code order). Half-widths
in grid units at node i of a step with fl = floor((xbar_k - z_0)/h), cl = ceil((xbar_k - z_0)/h):
    L = max(am[i_lo] * (i - fl) + bm[i_lo], 48),   U = max(ap[i_hi] * (cl - i) + bp[i_hi], 48).
  old  (round 4, 400 actions): am = ap = 0, bm, bp in {48, 96, ..., 960}  -> (L, U) = (48 (i_lo + 1), 48 (i_hi + 1)) exactly;
  edge (plan §2, 100 actions): e in {1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4} sigma -> a = 1, b = 24 e (an integer number of grid
       units); e = inf -> a = 0, b = 48. Then lower edge index = min(fl - 24 e-, i - 48) (= xbar_k - e- sigma rounded down to a node,
       or r_i - 2 sigma), upper edge index = max(cl + 24 e+, i + 48) (rounded up), i.e. outward rounding.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R4 = os.path.join(EV, 'round4'); R7 = os.path.join(EV, 'round7')
for _p in (R4,): sys.path.insert(0, _p)
import numpy as np
from numba import njit
import evaluator_window as ew                       # frozen round 4 (read-only): constants, grid, kernel windows
dr = ew.dr

# --- round 13 change (1): register the inputs of round13/inputs13.json in memory (as evaluator_window does for P5514) -------
INPUTS13 = json.load(open(os.path.join(HERE, 'inputs13.json')))
def _numeric(v): return isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v)
REGISTERED13 = []
for _n, _e in INPUTS13.items():
    if _n in dr.CFG['inputs']:                         # U, N (config.json), P5514 (evaluator_window): taken from their sources
        if _numeric(_e.get('a')): assert [float(x) for x in _e['a']] == [float(x) for x in dr.CFG['inputs'][_n]['a']], _n
        if _numeric(_e.get('chi')): assert [float(x) for x in _e['chi']] == [float(x) for x in dr.CFG['inputs'][_n]['chi']], _n
        continue
    if _numeric(_e.get('a')): _a = [float(x) for x in _e['a']]
    elif _numeric(_e.get('p')): _a = [float(x) for x in np.sqrt(np.asarray(_e['p'], np.float64))]   # a = sqrt(p), double precision
    else: continue                                     # no numeric amplitudes yet (PAR: filled by Part B): not registered
    if _numeric(_e.get('chi')): _chi = [float(x) for x in _e['chi']]
    elif 'chi_rule' in _e:                              # SCH: chi_m = (-pi m (m - 1)/4) mod 2 pi, m = 1..4
        _m = np.arange(1, 5, dtype=np.float64); _chi = [float(x) for x in np.mod(-np.pi*_m*(_m-1)/4, 2*np.pi)]
        assert np.allclose(_chi, [0.0, 1.5*np.pi, 0.5*np.pi, np.pi], rtol=0, atol=1e-15), _chi
    else: raise ValueError(f"inputs13.json: input {_n} has no phases")
    dr.CFG['inputs'][_n] = dict(a=_a, chi=_chi); REGISTERED13.append(_n)
for _n, _e in dr.CFG['inputs'].items():
    assert abs(float(np.sum(np.asarray(_e['a'], np.float64)**2))-1.0) <= 1e-12, (_n, _e['a'])

s, D, K, B = ew.s, ew.D, ew.K, ew.B_BUDGET
assert B == 316.8921168493721
NAME = os.environ.get('R13_INPUT', 'U')               # round 13 change (2)
assert NAME in dr.CFG['inputs'], NAME
MIN_HW = 48                                          # 2 sigma in sigma/24 grid units
X_PAD = 960                                          # padding of the extended index range (>= largest edge overshoot)
TOL24, TOL120 = 1e-11, 1e-10                         # conservation tolerances (DECISION 11-5; TOL24 raised from 1e-12 by round 13 DECISION 13-30)
E_SIGMA = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, np.inf)
P = 9


class AssertionStop(RuntimeError):
    """conservation / non-negativity / support-boundary assertion failed: the run stops (no fallback)."""


# --- the design grid (frozen) -------------------------------------------------------------------------------------------
XK, HK, Z, DZ, KA, KB = ew.design_grid(NAME)         # KA, KB: the frozen 88-sigma windows = kernel windows of every step
G = len(Z); Z0 = float(Z[0])
UK = (XK-Z0)/DZ
FL = np.floor(UK).astype(np.int64); CL = np.ceil(UK).astype(np.int64)
# round 13 change (4): input-specific assertions (the support argument assumes them for every input)
assert float(np.abs(np.diff(XK)).max()) <= 4.6*s, (NAME, float(np.abs(np.diff(XK)).max()/s))
assert int((KA[1:]-ew.X24).min()) >= 0 and int((KB[1:]+ew.X24).max()) <= G-1, NAME


# --- action sets --------------------------------------------------------------------------------------------------------
class ActionSet:
    def __init__(self, name, am, bm, ap, bp, lab_lo, lab_hi):
        self.name = name
        self.am, self.bm = np.asarray(am, np.int64), np.asarray(bm, np.int64)
        self.ap, self.bp = np.asarray(ap, np.int64), np.asarray(bp, np.int64)
        self.NM, self.NP = len(self.am), len(self.ap); self.n = self.NM*self.NP
        self.lab_lo, self.lab_hi = list(lab_lo), list(lab_hi)
        assert self.n <= 0xFFFF

    def labels(self):
        return [f"({a}, {b})" for a in self.lab_lo for b in self.lab_hi]

    def arrays(self):
        return dict(am=self.am, bm=self.bm, ap=self.ap, bp=self.bp, lab_lo=np.array(self.lab_lo), lab_hi=np.array(self.lab_hi),
                    name=np.array(self.name))

def old_set():
    """round 4 action set: (M-, M+) in {48, ..., 960}^2, code = i_- * 20 + i_+ (DECISION B-2 of round 4)."""
    b = [ew.STEP_M*(i+1) for i in range(ew.N_ACT1)]
    lab = [f"{v // 24}sigma" for v in b]
    return ActionSet('old', [0]*ew.N_ACT1, b, [0]*ew.N_ACT1, b, lab, lab)

def edge_set():
    """plan §2: (e-, e+) in E^2; finite e -> edge at xbar_k -/+ e sigma (rounded outward), at least 2 sigma from r; e = inf ->
    r -/+ 2 sigma only."""
    a, b, lab = [], [], []
    for e in E_SIGMA:
        if np.isinf(e):
            a.append(0); b.append(MIN_HW); lab.append('inf')
        else:
            n = 24*e; assert n == int(n); a.append(1); b.append(int(n)); lab.append(f"{e:g}")
    return ActionSet('edge', a, b, a, b, lab, lab)


@njit(cache=True)
def _rows(code, am, bm, ap, bp, NP, fl, cl, Lout, Uout):
    """half-widths (grid units) of every node of the full grid for one step's code row."""
    for i in range(code.size):
        c = np.int64(code[i]); im = c//NP; ip = c-im*NP
        Lm = am[im]*(i-fl)+bm[im]
        if Lm < 48: Lm = 48
        Up = ap[ip]*(cl-i)+bp[ip]
        if Up < 48: Up = 48
        Lout[i] = Lm; Uout[i] = Up


class Table11:
    """a round-11 table: codes[(K-1), G] uint16 (row r <-> step k = r + 2, array index p = r + 1) on the full sigma/24 grid."""
    def __init__(self, codes, acts, tag=''):
        assert codes.shape == (K-1, G) and codes.dtype == np.uint16
        self.codes, self.acts, self.tag = codes, acts, tag
        self.X = X_PAD
        self._L = np.empty(G, np.int64); self._U = np.empty(G, np.int64)

    def rows_full(self, p, out=None):
        L, U = (self._L, self._U) if out is None else out
        a = self.acts
        _rows(self.codes[p-1], a.am, a.bm, a.ap, a.bp, a.NP, FL[p], CL[p], L, U)
        return L, U

    def rows(self, p, a, b):
        L, U = self.rows_full(p)
        return L[a:b+1].copy(), U[a:b+1].copy()

    def is_sod(self): return False


def from_round4(codes_w, start, width, acts):
    """the round-4 window table (codes on [start, start + width - 1] per step) as a full-grid Table11: nodes outside the stored
    range take the boundary node's code (the frozen runtime lookup rule of round5/robustness.LookupTable); they carry zero mass
    under the round-4 table, so the forward is unaffected."""
    codes = np.empty((K-1, G), np.uint16)
    for r in range(K-1):
        s0, w = int(start[r]), int(width[r])
        idx = np.clip(np.arange(G), s0, s0+w-1)-s0
        codes[r] = codes_w[r, idx]
    return Table11(codes, acts, 'round4')


# --- numba kernels: copies of the frozen formulas on the full grid ------------------------------------------------------
@njit(cache=True)
def _tails_full(gh, ka, kb, X, gE, LgE, SgE):
    """extended arrays over [-X, G-1+X] (index e = j + X) of the kernel gh restricted to its window [ka, kb] (zero elsewhere):
    gE[e] = gh_j, LgE[e] = sum_{e' < e} gE (one prefix pass), SgE[e] = sum_{e' > e} gE (one suffix pass)."""
    ng = gh.size; L = ng+2*X
    for e in range(L):
        j = e-X
        gE[e] = gh[j] if (j >= ka and j <= kb) else 0.0
    LgE[0] = 0.0
    for e in range(1, L): LgE[e] = LgE[e-1]+gE[e-1]
    SgE[L-1] = 0.0
    for e in range(L-2, -1, -1): SgE[e] = SgE[e+1]+gE[e+1]

@njit(cache=True)
def _step_full(m, mn, gh, Mm, Mp, X, gE, LgE, SgE, Bl, Bu, SBl, PBu, Ph, qn):
    """one forward step on the full grid (copy of evaluator_window._gen_step with [a, b] = [0, G-1]); qn <- qhat per node.
    Returns (sum_i m_i qhat_i, sum mn, min mn, first and last j with mn_j != 0, number of non-zero mn_j,
    boundary ok: mn_0 == mn_{G-1} == 0, edges ok: every edge inside [-X, G-1+X])."""
    ng = m.size; base = -X
    nb = ng+2                                            # bucket index t = j + 1, j in [-1, G]
    for t in range(nb): Bl[t] = 0.0; Bu[t] = 0.0
    en = 0.0; eok = True
    for i in range(ng):
        l = i-Mm[i]; u = i+Mp[i]
        if l < -X or u > ng-1+X:
            eok = False; l = max(l, -X); u = min(u, ng-1+X)
        el = l-base; eu = u-base
        half = 0.5*(gE[el]+gE[eu])
        q = (LgE[el]+SgE[eu])+half
        Ph[i] = (LgE[eu]-LgE[el+1])+half
        qn[i] = q
        en += m[i]*q
        lc = min(max(l, -1), ng); uc = min(max(u, -1), ng)
        Bl[lc+1] += m[i]; Bu[uc+1] += m[i]
    SBl[nb] = 0.0
    for t in range(nb-1, -1, -1): SBl[t] = SBl[t+1]+Bl[t]  # SBl[t] = sum_{t' >= t} Bl
    PBu[0] = 0.0
    for t in range(1, nb): PBu[t] = PBu[t-1]+Bu[t-1]        # PBu[t] = sum_{t' < t} Bu
    for j in range(ng):
        t = j+1
        T = (SBl[t+1]+PBu[t])+0.5*(Bl[t]+Bu[t])
        mn[j] = m[j]*Ph[j]+gh[j]*T
    tot = 0.0; mmin = mn[0]; slo = -1; shi = -1; nnz = 0
    for j in range(ng):
        v = mn[j]; tot += v
        if v < mmin: mmin = v
        if v != 0.0:
            nnz += 1
            if slo < 0: slo = j
            shi = j
    ok = (mn[0] == 0.0) and (mn[ng-1] == 0.0)
    return en, tot, mmin, slo, shi, nnz, ok, eok

@njit(cache=True)
def _back_full(Vn, gE, LgE, SgE, Mm, Mp, X, rc, wq, gvE, LgvE, SgvE, VnE, Vout):
    """backward fixed-policy step on the full grid (copy of evaluator_window._gen_back): Vout[o] = rc[o] + wq qhat +
    Phat V_{k+1}(o) + left tail + right tail of gv = ghat V_{k+1}; V_{k+1} = Vn on [0, G-1], 0 on the padding."""
    ng = Vn.size; L = ng+2*X
    for e in range(L):
        j = e-X
        VnE[e] = Vn[j] if (j >= 0 and j < ng) else 0.0
        gvE[e] = gE[e]*VnE[e]
    LgvE[0] = 0.0
    for e in range(1, L): LgvE[e] = LgvE[e-1]+gvE[e-1]
    SgvE[L-1] = 0.0
    for e in range(L-2, -1, -1): SgvE[e] = SgvE[e+1]+gvE[e+1]
    for o in range(ng):
        el = o-Mm[o]+X; eu = o+Mp[o]+X; ei = o+X
        half = 0.5*(gE[el]+gE[eu])
        q = (LgE[el]+SgE[eu])+half
        Pp = (LgE[eu]-LgE[el+1])+half
        snd = (LgvE[el]+SgvE[eu])+0.5*(gvE[el]+gvE[eu])
        Vout[o] = rc[o]+wq*q+Pp*VnE[ei]+snd


def warmup():
    ew._warmup()
    ng = 64; X = 4; L = ng+2*X
    m = np.zeros(ng); m[20:40] = 0.05; mn = np.zeros(ng); gh = np.zeros(ng); gh[10:50] = 1/40
    Mm = np.full(ng, 3, np.int64); Mp = np.full(ng, 2, np.int64)
    gE, LgE, SgE = (np.empty(L) for _ in range(3))
    _tails_full(gh, 10, 49, X, gE, LgE, SgE)
    _step_full(m, mn, gh, Mm, Mp, X, gE, LgE, SgE, *[np.empty(ng+3) for _ in range(4)], np.empty(ng), np.empty(ng))
    _back_full(np.zeros(ng), gE, LgE, SgE, Mm, Mp, X, np.zeros(ng), 1.0, *[np.empty(L) for _ in range(4)], np.empty(ng))
    a = edge_set(); c = np.zeros(ng, np.uint16); _rows(c, a.am, a.bm, a.ap, a.bp, a.NP, 3, 4, np.empty(ng, np.int64), np.empty(ng, np.int64))


# --- forward and backward on a full grid --------------------------------------------------------------------------------
def forward(z, dz, xk, ka, kb, rows, X, tol, gam=None, stats=None, window=None):
    """forward on all nodes z (z_i = z_0 + i dz) at every step; rows(p) -> (L, U) int64 over the full grid; kernel of step p on
    [ka[p], kb[p]] (asserted to end in exact zeros). stats(p, m_k, qhat, L, U, lo, hi, c) is called once per step k >= 2 with the
    pre-update mass (read-only views). window = (wa, wb): additionally assert that the mass is exactly zero outside [wa[p], wb[p]]
    at every array index p (the round-4 support windows; used for every old-action-set table, clarification (3))."""
    ng = z.size; Kn = xk.size; L_ = ng+2*X
    gE, LgE, SgE = (np.empty(L_) for _ in range(3))
    Bl, Bu, SBl, PBu = (np.empty(ng+3) for _ in range(4)); Ph = np.empty(ng); qn = np.empty(ng)
    m = np.zeros(ng); mn = np.zeros(ng); gh = np.zeros(ng)
    EN = 1.0; ENP = 1.0; W = np.zeros(Kn); W[0] = 1.0; Q = np.zeros(Kn); Q[0] = 1.0
    mass = np.empty(Kn); minm = np.empty(Kn); kmass = np.empty(Kn)
    nnz = np.zeros(Kn, np.int64); slo = np.zeros(Kn, np.int64); shi = np.zeros(Kn, np.int64)
    g1 = np.empty(ng); kmass[0] = dr._kernel(z, xk[0], s, dz, g1)
    m[:] = g1                                                   # m_2 = ghat_1 (the full-grid kernel of step 1, as frozen)
    if not (m[0] == 0.0 and m[-1] == 0.0): raise AssertionStop("initialisation: non-zero mass at a grid end")
    nz = np.flatnonzero(m); lo, hi = int(nz[0]), int(nz[-1])
    mass[0] = m.sum(); minm[0] = m.min(); nnz[0] = nz.size; slo[0], shi[0] = lo, hi
    if window is not None and not (lo >= window[0][1] and hi <= window[1][1]):
        raise AssertionStop(f"m_2: non-zero mass outside the round-4 support window [{window[0][1]}, {window[1][1]}]")
    for p in range(1, Kn):
        A, Bb = int(ka[p]), int(kb[p])
        Mm, Mp = rows(p)
        zz = z[lo:hi+1]; mm = m[lo:hi+1]
        al = (zz-Mm[lo:hi+1]*dz-xk[p])/s; be = (zz+Mp[lo:hi+1]*dz-xk[p])/s
        cev, cno = dr.cfun(al, be); c = cev+cno
        W[p] = mm@c
        Pex = np.where(al > 0, dr.ndtr(-al)-dr.ndtr(-be), dr.ndtr(be)-dr.ndtr(al))
        ENP += mm@(1-Pex)
        gh[int(ka[p-1]):int(kb[p-1])+1] = 0.0
        kmass[p] = dr._kernel(z[A:Bb+1], xk[p], s, dz, gh[A:Bb+1])
        if not (gh[A] == 0.0 and gh[Bb] == 0.0):
            raise AssertionStop(f"step {p+1}: kernel non-zero at the end of its window [{A}, {Bb}]")
        _tails_full(gh, A, Bb, X, gE, LgE, SgE)
        en, mass[p], minm[p], lo_n, hi_n, nnz[p], ok, eok = _step_full(m, mn, gh, Mm, Mp, X, gE, LgE, SgE, Bl, Bu, SBl, PBu, Ph, qn)
        if not eok: raise AssertionStop(f"step {p+1}: a window edge outside the padded index range [-{X}, G-1+{X}]")
        if not ok: raise AssertionStop(f"step {p+1}: non-zero mass at the first or last grid node")
        if abs(mass[p]-1) > tol or minm[p] < 0:
            raise AssertionStop(f"step {p+1}: |sum m - 1| = {abs(mass[p]-1)!r} (tol {tol}) or min m = {minm[p]!r}")
        if stats is not None: stats(p, m, qn, Mm, Mp, lo, hi, c)
        if window is not None:
            pn = p+1 if p < Kn-1 else p
            if not (lo_n >= window[0][pn] and hi_n <= window[1][pn]):
                raise AssertionStop(f"step {p+1}: non-zero mass on a new support node, outside the round-4 support window "
                                    f"[{window[0][pn]}, {window[1][pn]}] (support [{lo_n}, {hi_n}])")
        slo[p], shi[p] = lo_n, hi_n
        Q[p] = en; EN += en; m, mn = mn, m; lo, hi = int(lo_n), int(hi_n)
    out = dict(EN=EN, ENP=ENP, W=W, Q=Q, mass=mass, minm=minm, kmass=kmass, G=ng, nnz=nnz, sup_lo=slo, sup_hi=shi,
               mass_dev_max=float(np.abs(mass-1).max()), min_m=float(minm.min()))
    if gam is not None: out['Fgam'] = float(np.asarray(gam)[1:]@W[1:])
    return out

def backward(table, gam):
    """fixed-policy backward on the full sigma/24 grid: (sum ghat_1 V^c_2, sum ghat_1 V^q_2) with V^c for sum_{k>=2} gam_k w_k
    and V^q for N = sum_{k>=2} E[q_k] (check 2)."""
    X = X_PAD; L_ = G+2*X
    gE, LgE, SgE, gvE, LgvE, SgvE, VnE = (np.empty(L_) for _ in range(7))
    gh = np.zeros(G); Vc = np.zeros(G); Vq = np.zeros(G); Vc_n = np.empty(G); Vq_n = np.empty(G); zq = np.zeros(G)
    for p in range(K-1, 0, -1):
        A, Bb = int(KA[p]), int(KB[p])
        Mm, Mp = table.rows_full(p)
        al = (Z-Mm*DZ-XK[p])/s; be = (Z+Mp*DZ-XK[p])/s
        cev, cno = dr.cfun(al, be); rc = gam[p]*(cev+cno)
        gh[:] = 0.0; dr._kernel(Z[A:Bb+1], XK[p], s, DZ, gh[A:Bb+1])
        _tails_full(gh, A, Bb, X, gE, LgE, SgE)
        _back_full(Vc, gE, LgE, SgE, Mm, Mp, X, rc, 0.0, gvE, LgvE, SgvE, VnE, Vc_n)
        _back_full(Vq, gE, LgE, SgE, Mm, Mp, X, zq, 1.0, gvE, LgvE, SgvE, VnE, Vq_n)
        Vc, Vc_n = Vc_n, Vc; Vq, Vq_n = Vq_n, Vq
    g1 = np.empty(G); dr._kernel(Z, XK[0], s, DZ, g1)
    a1, b1 = int(KA[1]), int(KB[1])
    return dict(Fc=float(g1[a1:b1+1]@Vc[a1:b1+1]), Nq=float(g1[a1:b1+1]@Vq[a1:b1+1]))


def record(fw):
    """record and J exactly as density_recursion_v1.evaluate (J = (H w)^T H / sigma^2, slogdet, S = sum w)."""
    J = (HK*fw['W'][:, None]).T@HK/s**2
    sgn, ld = np.linalg.slogdet(J)
    return dict(EN=float(fw['EN']), EN_P=float(fw['ENP']), S=float(fw['W'].sum()), logdet=float(ld), logdet_sign=float(sgn),
                mass_dev_max=fw['mass_dev_max'], min_m=fw['min_m']), J

def evaluate(table, gam=None, stats=None, window=None):
    """sigma/24 design-grid evaluation of a Table11 (or any object with rows_full(p)); window=True asserts zero mass outside the
    round-4 support windows [KA, KB] at every step."""
    fw = forward(Z, DZ, XK, KA, KB, table.rows_full, X_PAD, TOL24, gam=gam, stats=stats, window=((KA, KB) if window else None))
    rec, J = record(fw)
    return rec, J, fw


# --- sigma/120 re-evaluation (the frozen round-7 convention on the full grid) ---------------------------------------------
_FINE = {}
def fine_grid(f):
    """round 7's quadrature grid of factor f (dr.make_grid of the nominal output at n = 24 f, z_0 bitwise the coarse z_0), its
    88-sigma kernel windows, and the coarse cell of every node (round7/evaluator_refine.cell_index, frozen rule)."""
    if f not in _FINE:
        sys.path.insert(0, R7)
        import evaluator_refine as er                         # frozen round 7 (read-only)
        zf, dzf = dr.make_grid(XK, 24*f, 8*D, 'B'); Gf = len(zf)
        assert float(zf[0]) == Z0
        kaf, kbf = ew.windows88(XK, zf[0], dzf, Gf)
        cell, gap = er.cell_index(zf, z0=Z0, h=DZ)           # round 13 change (3): this input's own coarse origin and step
        assert int(cell.min()) >= 0 and int(cell.max()) <= G-1
        _FINE[f] = dict(z=zf, dz=dzf, G=Gf, ka=kaf, kb=kbf, cell=cell, gap=gap)
    return _FINE[f]

def evaluate_refined(table, f=5):
    """the frozen continuous policy (round 7 §1.1-§1.2: half-widths constant within the coarse cell, physical half-widths
    f L, f U fine steps, window centred at the fine node's own r) on the full fine grid; tolerance 1e-10 (DECISION 11-5)."""
    g = fine_grid(f); cell = g['cell']
    Lf = np.empty(g['G'], np.int64); Uf = np.empty(g['G'], np.int64)
    def rows(p):
        L, U = table.rows_full(p)
        np.multiply(L[cell], f, out=Lf); np.multiply(U[cell], f, out=Uf)
        return Lf, Uf
    t0 = time.perf_counter()
    fw = forward(g['z'], g['dz'], XK, g['ka'], g['kb'], rows, f*X_PAD, TOL120)
    rec, J = record(fw); rec['G'] = g['G']; rec['f'] = f; rec['wall_s'] = time.perf_counter()-t0
    return rec, J, fw


# --- shape and offset-binned statistics --------------------------------------------------------------------------------
OFF_BINS = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, np.inf)
BIN_LABELS = ('[0, 2)', '[2, 5)', '[5, 10)', '[10, 20)', '[20, 40)', '[40, inf)')

class Stats:
    """per-step accumulation over the nodes that carry mass (k >= 2): facing-edge offset (round4/part_c.shape convention: near side
    = the side of xbar_k as seen from r, edge = near-side half-width - |r - xbar_k|, sigma units, positive = xbar_k inside the
    window), and by |r - xbar_k| bin: mass, transmissions sum m qhat, information sum m c and sum gamma_k m c."""
    def __init__(self, gam):
        self.gam = gam; nb = len(OFF_BINS)-1
        self.mass = np.zeros(nb); self.send = np.zeros(nb); self.info = np.zeros(nb); self.ginfo = np.zeros(nb)
        self.near = np.zeros(nb); self.far = np.zeros(nb); self.edge_acc = np.zeros(nb)
        self.ev = []; self.ew = []; self.sendQ = 0.0
        self.edge_lo = []; self.edge_hi = []

    def __call__(self, p, m, qn, Mm, Mp, lo, hi, c):
        mm = m[lo:hi+1]; sel = mm > 0
        off = (Z[lo:hi+1]-XK[p])/s
        dlo = Mm[lo:hi+1]/24.0; dhi = Mp[lo:hi+1]/24.0
        near = np.where(off > 0, dlo, dhi); far = np.where(off > 0, dhi, dlo)
        edge = near-np.abs(off)
        bi = np.clip(np.searchsorted(OFF_BINS, np.abs(off), side='right')-1, 0, len(OFF_BINS)-2)
        nb = len(OFF_BINS)-1
        snd = mm*qn[lo:hi+1]
        self.mass += np.bincount(bi, mm, nb); self.send += np.bincount(bi, snd, nb)
        self.info += np.bincount(bi, mm*c, nb); self.ginfo += np.bincount(bi, self.gam[p]*mm*c, nb)
        self.near += np.bincount(bi, mm*near, nb); self.far += np.bincount(bi, mm*far, nb); self.edge_acc += np.bincount(bi, mm*edge, nb)
        self.ev.append(edge[sel].astype(np.float64)); self.ew.append(mm[sel].copy())
        self.sendQ += float(snd.sum())

    def result(self):
        ev = np.concatenate(self.ev); w = np.concatenate(self.ew); o = np.argsort(ev, kind='stable'); cw = np.cumsum(w[o])/w.sum()
        qs = (0.1, 0.25, 0.5, 0.75, 0.9)
        eq = [float(ev[o][np.searchsorted(cw, q)]) for q in qs]
        tot_send = float(self.send.sum())
        bins = []
        for j, lab in enumerate(BIN_LABELS):
            bins.append(dict(bin=lab, mass_share=float(self.mass[j]/(K-1)), sends=float(self.send[j]), send_share=float(self.send[j]/tot_send),
                             info=float(self.info[j]), gamma_info=float(self.ginfo[j]),
                             info_per_send=(float(self.info[j]/self.send[j]) if self.send[j] > 0 else None),
                             gamma_info_per_send=(float(self.ginfo[j]/self.send[j]) if self.send[j] > 0 else None),
                             near=(float(self.near[j]/self.mass[j]) if self.mass[j] > 0 else None),
                             far=(float(self.far[j]/self.mass[j]) if self.mass[j] > 0 else None),
                             edge=(float(self.edge_acc[j]/self.mass[j]) if self.mass[j] > 0 else None)))
        return dict(edge_quantiles=eq, edge_q10=eq[0], edge_median=eq[2], edge_q90=eq[4], edge_mean=float(w@ev/w.sum()),
                    bins=bins, body_send_share=bins[-1]['send_share'], body_info_per_send=bins[-1]['info_per_send'],
                    body_gamma_info_per_send=bins[-1]['gamma_info_per_send'], sends_total=tot_send), ev, w


# --- Monte Carlo at the runtime lookup rule (round4/part_c._mc_batch with the generic action map) ------------------------
def _mc_batch(args):
    codes_path, acts_d, R, seed = args
    z0 = Z0; dz = DZ
    rng = np.random.default_rng(seed)
    codes = np.load(codes_path, mmap_mode='r')
    am, bm, ap, bp, NP = acts_d['am'], acts_d['bm'], acts_d['ap'], acts_d['bp'], int(acts_d['NP'])
    r = XK[0]+s*rng.standard_normal(R); N = np.ones(R); Wk = np.zeros(K); Wk[0] = float(R); S = np.ones(R)
    nclip = 0
    for p in range(1, K):
        i = np.ceil((r-z0)/dz-0.5).astype(np.int64)                     # nearest node, ties -> smaller index
        nclip += int(((i < 0) | (i > G-1)).sum()); i = np.clip(i, 0, G-1)
        cd = np.asarray(codes[p-1][i], np.int64); im = cd//NP; ip = cd-im*NP
        Lm = np.maximum(am[im]*(i-FL[p])+bm[im], MIN_HW); Up = np.maximum(ap[ip]*(CL[p]-i)+bp[ip], MIN_HW)
        lo = r-Lm*dz; hi = r+Up*dz
        cev, cno = dr.cfun((lo-XK[p])/s, (hi-XK[p])/s); c = cev+cno
        Wk[p] = c.sum(); S += c
        zk = XK[p]+s*rng.standard_normal(R)
        snd = (zk <= lo) | (zk >= hi)
        N += snd; r = np.where(snd, zk, r)
    return Wk, N, S, nclip

def monte_carlo(codes_path, acts, rec, R=10000, batches=20, seed=20261109, workers=4):
    from multiprocessing import Pool
    ad = dict(am=acts.am, bm=acts.bm, ap=acts.ap, bp=acts.bp, NP=acts.NP)
    jobs = [(codes_path, ad, R//batches, seed+b) for b in range(batches)]
    with Pool(workers) as pool: outs = pool.map(_mc_batch, jobs)
    Ns = np.concatenate([x[1] for x in outs]); Ss = np.concatenate([x[2] for x in outs]); Wk = sum(x[0] for x in outs)/R
    J = (HK*Wk[:, None]).T@HK/s**2; ld = float(np.linalg.slogdet(J)[1])
    ldb = np.array([np.linalg.slogdet((HK*(x[0]/(R//batches))[:, None]).T@HK/s**2)[1] for x in outs])
    d = dict(R=R, batches=batches, seed=seed, EN_mc=float(Ns.mean()), EN_se=float(Ns.std(ddof=1)/np.sqrt(R)), S_mc=float(Ss.mean()),
             S_se=float(Ss.std(ddof=1)/np.sqrt(R)), logdet_mc=ld, logdet_batch_se=float(ldb.std(ddof=1)/np.sqrt(batches)),
             EN_eval=rec['EN'], S_eval=rec['S'], logdet_eval=rec['logdet'], n_clipped=int(sum(x[3] for x in outs)))
    d['z_EN'] = (d['EN_eval']-d['EN_mc'])/d['EN_se']; d['z_S'] = (d['S_eval']-d['S_mc'])/d['S_se']
    d['z_logdet'] = (d['logdet_eval']-d['logdet_mc'])/d['logdet_batch_se']
    return d


# --- check 5: half-widths and edges -------------------------------------------------------------------------------------
def check_edges(table):
    """L, U >= 48 at every (k, i) of the full grid; window edges are integer node indices by construction; for the edge set the
    xbar-based candidate edges are the nodes at or beyond xbar_k -/+ e sigma within one grid step (outward rounding)."""
    acts = table.acts; minL = 10**9; minU = 10**9; bad = 0
    L = np.empty(G, np.int64); U = np.empty(G, np.int64)
    for p in range(1, K):
        table.rows_full(p, (L, U)); minL = min(minL, int(L.min())); minU = min(minU, int(U.min()))
    worst_lo = 0.0; worst_hi = 0.0; viol = 0
    for im in range(acts.NM):
        if acts.am[im] == 0: continue
        e = acts.bm[im]/24.0
        zl = Z0+(FL-acts.bm[im])*DZ; tl = XK-e*s                  # candidate lower edge node and its target
        d = (tl-zl)/DZ; viol += int(((d < -1e-6) | (d >= 1+1e-6)).sum()); worst_lo = max(worst_lo, float(np.abs(d).max()))
    for ip in range(acts.NP):
        if acts.ap[ip] == 0: continue
        e = acts.bp[ip]/24.0
        zu = Z0+(CL+acts.bp[ip])*DZ; tu = XK+e*s
        d = (zu-tu)/DZ; viol += int(((d < -1e-6) | (d >= 1+1e-6)).sum()); worst_hi = max(worst_hi, float(np.abs(d).max()))
    return dict(min_L=minL, min_U=minU, min_ge_48=bool(minL >= 48 and minU >= 48), outward_rounding_violations=viol,
                max_rounding_lo_grid=worst_lo, max_rounding_hi_grid=worst_hi, edges_on_nodes=True,
                pass_=bool(minL >= 48 and minU >= 48 and viol == 0))
