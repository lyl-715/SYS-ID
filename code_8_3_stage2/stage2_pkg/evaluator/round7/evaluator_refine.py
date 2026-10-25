"""Round 7, Part A1: refinement of the general-window evaluator (spec reports_CN/plan_round7_CN.md v3.1, frozen; §1.2, §3
items 1 and 3). 固定设计、离散方案下 -- a finer quadrature of the SAME frozen continuous policy; no redesign, not a bound.

The round-4 general window path (round4/evaluator_window.py: forward, backward, windows88, ConstTable; the accepted
operator evaluator_fast / density_recursion_v1 version B) and the round-5 mismatch conventions (round5/robustness.py:
x_of, patched, LookupTable, the frozen nominal grid) are imported unchanged. This module adds only:

  * frozen absolute coordinates: z_0 = min xbar - 8 Delta and h_24 = sigma/24 of the nominal sigma/24 grid (dr.make_grid of
    the nominal output; round 4/5/6), every table's per-step support start index and width on that grid;
  * the quadrature grid of refinement factor f (step h_24/f): dr.make_grid(xbar, 24 f, 8 Delta, 'B') of the NOMINAL output,
    z_j = z_0 + j h_f (z_0 bitwise the coarse z_0; for f = 1 the coarse grid itself), optionally shifted by half a step,
    z_j = z_0 + (j + 1/2) h_f (Part B diagnosis only; coarse origin, cell boundaries and policy untouched);
  * the cell lookup (spec §1.1-§1.2): the frozen continuous policy is "half-widths constant within the cell of width h_24
    centred at each coarse node, window centre = the actual r". A quadrature node at r belongs to coarse node i iff
    |r - r_i| < h_24/2, computed from physical coordinates: u = (r - z_0)/h_24, i = floor(u + 1/2); the run stops
    (BoundaryNode) if any node has |u - i| > 1/2 - 1e-9, i.e. lies on (or within 1e-9 cell widths of) a cell boundary.
    A node whose cell lies outside the step's stored range takes the range's boundary node (round 5's rule, unchanged).
    Half-widths are physical: M^+- h_24 = f M^+- grid steps; the window is centred at the node's own r;
  * the dispatch (spec §3 item 1): a table whose every window is (480, 480) coarse units, i.e. (480 f, 480 f) on the grid
    -> the accepted operator at n = 24 f with Delta* fixed (evaluator_fast.evaluate_fast, bitwise equal to
    density_recursion_v1.evaluate('B', ., 24 f): report_speedup_CN.md), input arrays (x(theta), H), grid pinned to the
    nominal grid ("reuse path"); every other table, including every other constant window (f M, f M), -> the general
    window path (R = 88 sigma, support windows centred at x(theta) on the frozen grid; "general path"). M is always in
    coarse units; the physical thresholds do not depend on f.

Support half-width R = 88 sigma (physical), so the support window holds about 176 sigma / h_f nodes and the largest
half-width X = f * 960 grid steps; the conservation / non-negativity / support assertions of the round-4 forward apply
unchanged at every step (they are raised as evaluator_window.AssertionStop).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R4 = os.path.join(EV, 'round4'); R5 = os.path.join(EV, 'round5'); R6 = os.path.join(EV, 'round6')
for _p in (R4, R5, R6): sys.path.insert(0, _p)
import numpy as np
import robustness as rb                                   # round 5, unchanged
ew, ef, dr = rb.ew, rb.ef, rb.dr

s, D, K, B = rb.s, rb.D, rb.K, rb.B
assert B == 316.8921168493721
NAME = 'U'
N24 = ew.N_GRID; SOD_M = ew.SOD_M
FACTORS = (1, 3, 5)
CELL_TOL = 1e-9                                           # cell widths; |u - i| must stay below 1/2 - CELL_TOL
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
for _d in (LOGS, RES): os.makedirs(_d, exist_ok=True)
VERDICT = '离散方案下'

I24 = rb.INP[NAME]                                         # frozen nominal sigma/24 grid (z, dz, G) and xbar, H
Z0, H24, G24 = float(I24['z'][0]), float(I24['dz']), int(I24['G'])
assert Z0 == float(I24['xbar'].min()-8*D) and H24 == s/24


class BoundaryNode(RuntimeError):
    """a quadrature node lies on (or within CELL_TOL of) a coarse cell boundary: the cell lookup is undefined; run stops."""


# --- grids -------------------------------------------------------------------------------------------------------------
_GRIDS = {}
def grid(f, shift=False):
    """the frozen quadrature grid of factor f: dr.make_grid of the NOMINAL output at n = 24 f (nodes z_0 + j h_f); with
    shift=True the same count of nodes at z_0 + (j + 1/2) h_f. Returns dict(z, dz, G, f, shift, cell) with the cell index
    of every node (cell_index)."""
    key = (int(f), bool(shift))
    if key not in _GRIDS:
        z, dz = dr.make_grid(I24['xbar'], N24*f, 8*D, 'B')
        if shift: z = Z0+(np.arange(len(z))+0.5)*dz
        cell, dmin = cell_index(z)
        _GRIDS[key] = dict(z=z, dz=dz, G=len(z), f=int(f), shift=bool(shift), cell=cell, boundary_gap=dmin)
    return _GRIDS[key]

def cell_index(z, z0=None, h=None):
    """coarse cell of every node at physical position z: i = floor((z - z_0)/h_24 + 1/2), i.e. |z - z_i| < h_24/2 (coarse
    origin z0 and spacing h default to the frozen Z0, H24). Raises BoundaryNode if a node is within CELL_TOL cell widths of a
    cell boundary (|u - i| >= 1/2 - CELL_TOL). Returns (i, min over nodes of 1/2 - |u - i|), the smallest distance to a
    boundary in cell widths."""
    z0 = Z0 if z0 is None else z0; h = H24 if h is None else h
    u = (np.asarray(z, float)-z0)/h
    i = np.floor(u+0.5).astype(np.int64)
    gap = 0.5-np.abs(u-i)
    if not (gap > CELL_TOL).all():
        j = int(np.argmin(gap))
        raise BoundaryNode(f"node {j} at u = {u[j]!r} cell widths is {gap[j]!r} from a cell boundary (tolerance {CELL_TOL})")
    return i, float(gap.min())


# --- refined tables ---------------------------------------------------------------------------------------------------
def stored_range(tab, p):
    """closed coarse index range of the nodes the coarse table stores at array index p (None: every node, constant tables)."""
    if hasattr(tab, 'start') and hasattr(tab, 'width'):
        r = p-1; return int(tab.start[r]), int(tab.start[r]+tab.width[r]-1)
    if isinstance(tab, ew.FullTable): return 0, int(tab.mm.shape[1])-1
    return None

class RefinedTable:
    """the coarse table `tab` (rows(p, a, b) on coarse nodes, half-widths in coarse units) on the quadrature grid with cell
    indices `cell`: node j takes the coarse node cell[j] (clipped to the step's stored range), half-widths f M^+- (physical
    M^+- h_24), window centred at the node's own r (the general forward centres it at z_j)."""
    def __init__(self, tab, f, cell):
        self.c, self.f, self.cell = tab, int(f), cell; self.X = int(f)*int(tab.X)
    def rows(self, p, a, b):
        ic = self.cell[a:b+1]
        rg = stored_range(self.c, p)
        if rg is not None: ic = np.clip(ic, rg[0], rg[1])
        lo = int(ic.min()); mm, mp = self.c.rows(p, lo, int(ic.max()))
        o = ic-lo
        return self.f*mm[o], self.f*mp[o]
    def is_sod(self): return bool(self.c.is_sod())

class SodTable(ew.ConstTable):
    """constant window (M, M) in coarse units as a coarse table (is_sod() iff M = 480)."""
    def __init__(self, M): super().__init__(M, M)


# --- evaluation at a true theta -----------------------------------------------------------------------------------------
def x_theta(theta): return rb.x_of(NAME, np.asarray(theta, float))

def eval_reuse(f, xth):
    """the accepted operator at n = 24 f, Delta* fixed (every window (480 f, 480 f)): evaluator_fast.evaluate_fast with input
    arrays (x(theta), H) and its grid pinned to the nominal grid (round 5's eval_sod at f = 1, round 6's eval_sod48 at f = 2)."""
    mg = dr.make_grid
    with rb.patched(dr, input_arrays=lambda nm: (xth, I24['H']), make_grid=lambda xk, n, margin, version: mg(I24['xbar'], n, margin, version)):
        rec, J, r = ef.evaluate_fast(NAME, N24*f)
    rec['path'] = f'reuse: accepted operator (evaluator_fast.evaluate_fast, n = {N24*f}), nominal grid, windows centred at x(theta)'
    return rec, J, r

def eval_general(tab, f, xth, shift=False):
    """the general window path on the frozen grid of factor f (shifted if asked) with the refined table."""
    g = grid(f, shift); z, dz, G = g['z'], g['dz'], g['G']
    rt = RefinedTable(tab, f, g['cell'])
    a, b = ew.windows88(xth, z[0], dz, G)
    assert int((a-rt.X).min()) >= 0 and int((b+rt.X).max()) <= G-1, "a window would cross the frozen grid"
    def run(xk, n_, margin, version):
        assert xk is xth and version == 'B'
        return ew.forward(z, dz, s, xth, a, b, rt)
    with rb.patched(dr, input_arrays=lambda nm: (xth, I24['H']), run_ab=run):
        rec, J, r = dr.evaluate('B', NAME, N24*f, 8)
    rec['path'] = f"general window path (R = 88 sigma), f = {f}{' shifted by h_f/2' if shift else ''}, frozen grid, windows centred at x(theta)"
    # boundary lookups with mass: steps where the mass support (m_k at array index p from sup of p - 1) leaves the stored range
    rec['boundary_steps'] = 0
    if hasattr(tab, 'start'):
        lo, hi = r['sup_lo'][:-1], r['sup_hi'][:-1]
        cl, ch = g['cell'][lo], g['cell'][hi]; st, en = tab.start, tab.start+tab.width-1
        rec['boundary_steps'] = int(((cl < st) | (ch > en)).sum())
    return rec, J, r

def evaluate(tab, f, theta=None, xth=None, shift=False, force_general=False):
    """dispatch (spec §3 item 1): every window (480, 480) coarse units and an unshifted grid -> reuse path (accepted operator
    at n = 24 f); otherwise the general path. Returns (record, J, run dict); record['route'] in {'reuse', 'general'}."""
    if xth is None: xth = x_theta(theta)
    if tab.is_sod() and not shift and not force_general:
        rec, J, r = eval_reuse(f, xth); rec['route'] = 'reuse'
    else:
        rec, J, r = eval_general(tab, f, xth, shift); rec['route'] = 'general'
    rec['f'] = int(f); rec['shift'] = bool(shift); rec['assert_fail'] = int((~r['assert_ok']).sum())
    return rec, J, r

KEEP = ('EN', 'EN_P', 'S', 'logdet', 'logdet_sign', 'mass_dev_max', 'min_m', 'kernel_mass_dev_max', 'w_min', 'w_max',
        'inv_S', 'inv_J', 'inv_w', 'runtime_s', 'path', 'route', 'boundary_steps', 'assert_fail', 'G', 'f', 'shift')

def record(rec, J):
    d = {k: (rec[k].item() if hasattr(rec[k], 'item') else rec[k]) for k in KEEP if k in rec}
    d['J'] = np.asarray(J).tolist()
    return d


# --- tables -------------------------------------------------------------------------------------------------------------
TABLE_NPZ = {'m000': os.path.join(R4, 'results', 'table_U_W.npz'), 'm005': os.path.join(R6, 'results', 'table_U_W_m005.npz'),
             'm010': os.path.join(R6, 'results', 'table_U_W_m010.npz'), 'm020': os.path.join(R6, 'results', 'table_U_W_m020.npz'),
             'T4': os.path.join(RES, 'table_T4.npz')}
CANDIDATES = ('m000', 'm005', 'm010', 'm020', 'T4')

def load_table(tid):
    """a frozen DP table as round 5's LookupTable, its support start indices and widths checked against the nominal sigma/24
    grid's support windows (absolute coordinates z_0, h_24 are those of that grid)."""
    a, b = ew.windows88(I24['xbar'], Z0, H24, G24)
    with np.load(TABLE_NPZ[tid]) as f_:
        codes, start, width, lam, run = f_['codes'], f_['start'], f_['width'], float(f_['lam']), int(f_['run'])
    assert np.array_equal(start, a[1:]) and np.array_equal(width, (b-a+1)[1:]), tid
    t = rb.LookupTable(codes, start, width); t.lam, t.run, t.tid = lam, run, tid
    return t

def load_tables(ids=CANDIDATES): return {t: load_table(t) for t in ids}


# --- the 27 points ------------------------------------------------------------------------------------------------------
def points27():
    """theta_0 + the round-6 26-point selection set, exact theta vectors from round6/results/archive_partA.jsonl (rows of the
    m020 table), checked bitwise against round 5's points('U')."""
    A = [json.loads(l) for l in open(os.path.join(R6, 'results', 'archive_partA.jsonl'))]
    q = [r for r in A if r['table'] == 'm020' and (r['cls'] == 'theta0' or r['selection26'])]
    assert len(q) == 27
    p5 = {f"{p['cls']}|{p['dir']}|{'+' if p['sign'] > 0 else '-'}|{p['eps_target']}": p for p in rb.POINTS[NAME]}
    out = []
    for r in q:
        pid = 'theta_0' if r['cls'] == 'theta0' else f"{r['cls']}|{r['dir']}|{'+' if r['sign'] > 0 else '-'}|{r['eps_target']}"
        th = np.asarray(r['theta'], float)
        if pid == 'theta_0': assert np.array_equal(th, rb.THETA0)
        else: assert np.array_equal(th, p5[pid]['theta']), pid
        out.append(dict(pid=pid, cls=r['cls'], dir=r['dir'], sign=r['sign'], eps_target=r['eps_target'], theta=[float(v) for v in th],
                        sel26=r['cls'] != 'theta0'))
    return out

def label(pid):
    if pid == 'theta_0': return 'theta_0'
    c, d, sg, e = pid.split('|')
    return f"{c} {d} {sg}{e}" if c == 'P1' else f"{c} {d} {e}"


def warmup(): ew._warmup()
