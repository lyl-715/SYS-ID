"""Round 8: post-hoc evaluation at the true theta_0 (spec reports_CN/plan_round8_CN.md v3.1, frozen; §1.4, §2). theta_0
allowed here. Reuses round 7's refined evaluator (evaluator_refine: RefinedTable, cell_index, eval_reuse), round 5's
LookupTable and round 4's general window path / the accepted evaluator's record and J assembly, unchanged.

  E[N_2 | T, theta_0], J_table(T; theta_0): the general window path at the reference grid sigma/120 (f = 5) with the cell
  lookup of the frozen continuous policy. Difference to round 7 (whose tables all lived on the nominal grid): every table
  carries its OWN coarse origin z0 (generated from x_hat by the design), so the quadrature grid is anchored on that table's
  lattice: z_j = z0' + j h_f, z0' = z0 + n0 h_f with n0 = round((min x0 - 8 Delta - z0)/h_f), the node count G that
  dr.make_grid(x0, 24 f, 8 Delta) gives (round 7's grid of the true output, shifted by less than h_f/2 onto the table's
  lattice; for a table designed at x_hat = x0 it is round 7's grid bit for bit, n0 = 0). For odd f no node lies on a cell
  boundary (cell_index asserts a gap > 1e-9 cell widths); the support windows are round 4's (R = 88 sigma) centred at x0.
  b_1 = E_theta0[N_SoD], J_1: the reuse path at n = 120 (evaluate_fast at x0, nominal grid; round 7 eval_reuse(5, x0)).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
for _p in (HERE, os.path.join(EV, 'round7')):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np
import evaluator_refine as er                      # round 7, unchanged (imports round 5 robustness, round 4 evaluator_window)
import truth_r8 as tr
rb, ew, dr = er.rb, er.ew, er.dr
s, D, K = er.s, er.D, er.K
F_REF = 5                                          # sigma/120
assert np.array_equal(tr.X0, rb.INP['U']['xbar'])
THETA0 = tr.THETA0; X0 = tr.X0


def table_grid(z0_T, h_T, f=F_REF, x0=None):
    """quadrature grid of factor f anchored on the table's lattice (see module docstring); returns (z, dz, G, cell, gap, n0)."""
    x0 = X0 if x0 is None else x0
    zr, dz = dr.make_grid(x0, er.N24*f, 8*D, 'B'); G = len(zr)
    n0 = int(np.round((zr[0]-z0_T)/dz))
    z = (z0_T+n0*dz)+np.arange(G)*dz                          # n0 = 0 and z0_T = min x0 - 8 Delta: round 7's grid bit for bit
    cell, gap = er.cell_index(z, z0_T, h_T)
    return z, dz, G, cell, gap, n0

def eval_table(tab, f=F_REF, x0=None):
    """general window path at the true output for the SeqTable-like `tab` (codes, start, width, z0, h); returns (record, J,
    run dict) exactly as evaluator_refine.eval_general, plus the grid anchoring."""
    x0 = X0 if x0 is None else x0
    lt = rb.LookupTable(tab.codes, tab.start, tab.width)
    assert abs(tab.h-er.H24) == 0.0
    z, dz, G, cell, gap, n0 = table_grid(tab.z0, tab.h, f, x0)
    rt = er.RefinedTable(lt, f, cell)
    a, b = ew.windows88(x0, z[0], dz, G)
    assert int((a-rt.X).min()) >= 0 and int((b+rt.X).max()) <= G-1, "a window would cross the quadrature grid"
    def run(xk, n_, margin, version):
        assert xk is x0 and version == 'B'
        return ew.forward(z, dz, s, x0, a, b, rt)
    with rb.patched(dr, input_arrays=lambda nm: (x0, er.I24['H']), run_ab=run):
        rec, J, r = dr.evaluate('B', 'U', er.N24*f, 8)
    lo, hi = r['sup_lo'][:-1], r['sup_hi'][:-1]
    cl, ch = cell[lo], cell[hi]; st, en = lt.start, lt.start+lt.width-1
    rec.update(boundary_steps=int(((cl < st) | (ch > en)).sum()), grid_n0=n0, boundary_gap=gap, f=f,
               assert_fail=int((~r['assert_ok']).sum()),
               path=f'general window path (R = 88 sigma), f = {f}, grid anchored on the table lattice (n0 = {n0}), windows centred at x0')
    return rec, J, r

def eval_sod_ref(f=F_REF):
    """b_1 and J_1: constant Delta*, the same first-send rule, reuse path at n = 24 f (round 7 eval_reuse at x0)."""
    return er.eval_reuse(f, X0)

def warmup(): er.warmup()
