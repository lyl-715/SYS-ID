"""Round 8: the true system and the simulator (spec reports_CN/plan_round8_CN.md v3.1, frozen; §1, §1.1, §1.4, §1.6).

The ONLY places theta_0 lives are this module and the post-hoc evaluation (posthoc_r8.py). The design and estimation
modules never import it (isolation_r8.py checks the import graph and replaces THETA0 / X0 by sentinels while a design runs).

  THETA0  = [G(0) a0, Re G(j w_1), Im G(j w_1), ..., Re G(j w_4), Im G(j w_4)] (mc84.py expression, regressor layout of mc83.py)
  X0      = the nominal output of the uniform input, mc83.py setup block (bitwise density_recursion_v1.input_arrays('U')[0],
            the x(theta_0) of rounds 4-7; checked in checks_r8.py)

Simulator (both stages: forced first transmission, reference reset, the same H and clock):
  stage 1 and the control's stage 2: mc83.sod(z, Delta*) (the reused trigger: transmit iff |z_k - r| >= Delta*);
  sequential stage 2: sim_table(z, table) -- the table's window is looked up from the ACTUAL reference value r (cell of the
  table's own coarse grid, clipped to the step's stored range; tables_r8.SeqTable.lookup) and z_k is transmitted iff it lies
  outside the window, (z_k - r) >= M^+ h or (r - z_k) >= M^- h (no quadrature half weights). With every window (480, 480)
  this is mc83.sod's |z_k - r| >= Delta* exactly (480 h = Delta* in floating point and fl(r - z) = -fl(z - r); checked).
Noise: e^(1) = default_rng(20261028 + r).standard_normal(K), e^(2) = default_rng(20271028 + r).standard_normal(K); z = X0 + sigma e.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); PKG = os.path.dirname(EV)
if HERE not in sys.path: sys.path.insert(0, HERE)
import numpy as np

_ns = {}
_cwd = os.getcwd(); os.chdir(PKG)
try:
    exec(open('mc83.py').read().split('# --- noise-free')[0], _ns)     # x, H, K, t, w, G, Gj, a0, sod, cfun (setup block)
finally:
    os.chdir(_cwd)
D = 0.002384694482635785; s = D/20
K = _ns['K']
THETA0 = np.concatenate([[_ns['G'](0)*_ns['a0']], np.ravel(np.column_stack([_ns['Gj'].real, _ns['Gj'].imag]))])
X0 = _ns['x']
sod = _ns['sod']
H_MC83 = _ns['H']
SEED1, SEED2 = 20261028, 20271028


def noise(r):
    """the two pre-generated noise sequences of replication r (full length K)."""
    e1 = np.random.default_rng(SEED1+r).standard_normal(K)
    e2 = np.random.default_rng(SEED2+r).standard_normal(K)
    return e1, e2

def record_from(z, eps):
    """the observed record: transmitted flags and the transmitted values (non-transmitted entries NaN)."""
    zt = np.where(eps, z, np.nan)
    return dict(ev=np.asarray(eps, bool), zt=zt)

def sim_sod(e):
    """constant-Delta* SoD stage (mc83.sod) driven by noise e."""
    z = X0+s*e
    eps, r, _ = sod(z, D)
    return z, eps, r

def sim_table(e, table):
    """sequential stage 2 with the window table `table` (tables_r8.SeqTable) driven by noise e: forced first transmission,
    reference reset r = z_1, lookup from the actual reference, transmit iff z_k outside the open window."""
    z = X0+s*e
    eps = np.zeros(K, bool); r = np.empty(K); r[0] = np.nan; eps[0] = True
    last = z[0]; h = table.h
    for k in range(1, K):
        r[k] = last
        mm, mp = table.lookup(k+1, last)
        if z[k]-last >= mp*h or last-z[k] >= mm*h:
            eps[k] = True; last = z[k]
    return z, eps, r
