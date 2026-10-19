"""§8.5 evaluator: deterministic u -> (E[N], S, J) by the exact 1-D density recursion of the send-on-delta reference
value r_k, discretised on a node grid (spec reports_CN/plan_evaluator_CN.md v4.1, §2-§4, §6).

  --version v0 : the prototype ../density_recursion.py run(), arithmetic unchanged (numpy; regression reference only)
  --version A  : mass formulation (§4), omega_ij = 1{|i-j| < M}                       (numba)
  --version B  : mass formulation (§4), omega_ij = 1{|i-j| < M} + 1/2 1{|i-j| = M}    (numba)

Written once and shared by the three versions: the setup block (mc83.py via mc85_scan.py: x, H, cfun, D, s, K, t), the
input arrays (output/regressors exactly as stage2_common.evaluate_crn, so recursion and MC use the same x_k, h_k), the
grid, J assembly, the record of spec §6 and the invariant checks.

A/B step (§4): ghat_j = g(z_j)/sum_j' g(z_j'); Phat_i = sum_j ghat_j omega_ij; m'_j = m_j Phat_j + ghat_j sum_i m_i (1-omega_ij).
Window sums by prefix sums, O(G) per step. 1 - Phat_i and sum_i m_i (1-omega_ij) are formed directly as the sums over
|i-j| > M plus the weighted boundary terms (differences of a non-decreasing prefix array, hence >= 0 in floating point),
not as 1 - Phat. m is never renormalised. c_k = mc83.cfun (the reviewed implementation, reused, not ported) on the index
range where m_k != 0 (exact: m_k = 0 elsewhere).

Usage (from this directory):
  python3 density_recursion_v1.py --version B --input U --n 12 [--margin 8]
  python3 density_recursion_v1.py --test        # unit tests (a), (b), (c) of step 2 -> logs/unit_tests.log
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
_cwd = os.getcwd(); os.chdir(PKG)
# setup block of mc83.py + D, s, regressors(), output() exactly as in mc85_scan.py (same split as stage2_common.py)
exec(open('mc85_scan.py').read().split('def evaluate(a, chi):')[0])
del R, rng, Z                     # the stage-1 draw block defined there; the evaluator draws nothing
os.chdir(_cwd)
assert D == 0.002384694482635785 and s == D/20 and K == 5000
CFG = json.load(open(os.path.join(HERE, 'config.json')))
TWO_PI = 2*np.pi
RES = os.path.join(HERE, 'results'); LOGS = os.path.join(HERE, 'logs')
VERSIONS = ('v0', 'A', 'B'); WB = {'A': 0.0, 'B': 0.5}       # weight of the boundary |i-j| = M inside Phat
TAIL = 30.0                                                    # P12: tail nodes |alpha| > 30 and |beta| > 30


class Tee:
    """copy stdout to a log file (every number in the report is traceable to these logs)."""
    def __init__(self, path, mode='w'): self.f = open(path, mode); self.o = sys.stdout
    def write(self, m): self.o.write(m); self.f.write(m); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()


# --- shared: inputs, grid ------------------------------------------------------------------------------
def input_arrays(name):
    """x_k and H for a config input, built exactly as in stage2_common.evaluate_crn (chi reduced mod 2 pi)."""
    e = CFG['inputs'][name]
    a = np.asarray(e['a'], float); chi = np.mod(np.asarray(e['chi'], float), TWO_PI)
    return output(a, chi), regressors(a, chi, t)

def make_grid(xk, n, margin, version):
    """nodes on [min x - margin, max x + margin], h = sigma/n, G = len(np.arange(lo, hi+h, h)) as in the prototype.
    v0: the prototype's np.arange(lo, hi+h, h) (its step fl(lo+h)-lo differs from h by ~3e-12 relative);
    A, B: z_i = z_0 + i h exactly as spec §4 (z_0 = lo)."""
    lo, hi = xk.min()-margin, xk.max()+margin; dz = s/n
    z = np.arange(lo, hi+dz, dz)
    return (z if version == 'v0' else lo+np.arange(len(z))*dz), dz


# --- v0: the prototype, arithmetic unchanged --------------------------------------------------------------
def run_v0(xk, n, margin):
    """density_recursion.py run() lines 9-29 with x -> xk. Added records only (per-step mass, min m, kernel mass);
    E[N], W and the update of f are the prototype's expressions."""
    z, dz = make_grid(xk, n, margin, 'v0'); ng = len(z)
    f = np.exp(-0.5*((z-xk[0])/s)**2)/(s*np.sqrt(2*np.pi))        # f_2 = g_1
    EN = 1.0; W = np.zeros(K); W[0] = 1.0
    mass = np.empty(K); minm = np.empty(K); kmass = np.empty(K)
    mass[0] = np.sum(f)*dz; minm[0] = f.min()*dz; kmass[0] = mass[0]
    sh = int(round(D/dz))
    for k in range(1, K):
        xk_ = xk[k]; al = (z-D-xk_)/s; be = (z+D-xk_)/s
        P = ndtr(be)-ndtr(al)
        ph = lambda u: np.exp(-0.5*u*u)/np.sqrt(2*np.pi)
        cev = 1-P-al*ph(al)+be*ph(be); cno = np.where(P > 1e-300, (ph(al)-ph(be))**2/np.maximum(P, 1e-300), 0.0)
        W[k] = np.sum(f*(cev+cno))*dz; EN += np.sum(f*(1-P))*dz
        F = np.cumsum(f)*dz
        Fp = np.concatenate([F[sh:], np.full(sh, F[-1])])
        Fm = np.concatenate([np.zeros(sh), F[:-sh]])
        g = np.exp(-0.5*((z-xk_)/s)**2)/(s*np.sqrt(2*np.pi))
        f = f*P + g*(1-Fp+Fm)
        mass[k] = np.sum(f)*dz; minm[k] = f.min()*dz; kmass[k] = np.sum(g)*dz
    return dict(EN=EN, ENP=EN, W=W, mass=mass, minm=minm, kmass=kmass, G=ng, M=sh, Delta_over_h=D/dz)


# --- A / B: mass formulation (numba) ----------------------------------------------------------------------
@njit(cache=True)
def _kernel(z, xk, sg, dz, gh):
    """gh <- ghat_j = g(z_j)/sum_j' g(z_j'), g as in the prototype; returns sum_j g(z_j) h (truncated kernel mass)."""
    tot = 0.0; c0 = sg*np.sqrt(2*np.pi)
    for j in range(z.size):
        u = (z[j]-xk)/sg; g = np.exp(-0.5*u*u)/c0; gh[j] = g; tot += g
    for j in range(z.size): gh[j] = gh[j]/tot
    return tot*dz

@njit(cache=True)
def _prefix(v, C):
    C[0] = 0.0
    for j in range(v.size): C[j+1] = C[j]+v[j]

@njit(cache=True)
def _window(v, C, i, M, wb):
    """(sum_j v_j omega_ij, sum_j v_j (1-omega_ij)) for omega = 1{|i-j|<M} + wb 1{|i-j|=M}, from the prefix array C of v."""
    ng = v.size
    inside = C[min(i+M, ng)]-C[max(i-M+1, 0)]
    outside = C[max(i-M, 0)]+(C[ng]-C[min(i+M+1, ng)])
    b = 0.0
    if i-M >= 0: b += v[i-M]
    if i+M < ng: b += v[i+M]
    return inside+wb*b, outside+(1.0-wb)*b

@njit(cache=True)
def _ab_step(m, mn, gh, Cg, Cm, M, wb):
    """one step of spec §4. mn <- m_{k+1}. Returns sum_i m_i (1-Phat_i), sum mn, min mn, first and last j with mn_j != 0."""
    ng = m.size
    _prefix(gh, Cg); _prefix(m, Cm)
    en = 0.0
    for i in range(ng):
        pin, pout = _window(gh, Cg, i, M, wb)
        _, T = _window(m, Cm, i, M, wb)
        en += m[i]*pout
        mn[i] = m[i]*pin+gh[i]*T
    tot = 0.0; mmin = mn[0]; lo = -1; hi = -1
    for j in range(ng):
        v = mn[j]; tot += v
        if v < mmin: mmin = v
        if v != 0.0:
            if lo < 0: lo = j
            hi = j
    return en, tot, mmin, lo, hi

def run_ab(xk, n, margin, version):
    z, dz = make_grid(xk, n, margin, version); ng = len(z); M = 20*n; wb = WB[version]
    assert abs(D/dz-M) < 1e-6, (D/dz, M)
    m = np.empty(ng); mn = np.empty(ng); gh = np.empty(ng); Cg = np.empty(ng+1); Cm = np.empty(ng+1)
    EN = 1.0; ENP = 1.0; W = np.zeros(K); W[0] = 1.0
    mass = np.empty(K); minm = np.empty(K); kmass = np.empty(K)
    kmass[0] = _kernel(z, xk[0], s, dz, gh); m[:] = gh                      # m_2 = ghat_1
    nz = np.flatnonzero(m); lo, hi = nz[0], nz[-1]
    mass[0] = m.sum(); minm[0] = m.min()
    for k in range(1, K):
        zz = z[lo:hi+1]; mm = m[lo:hi+1]
        al = (zz-D-xk[k])/s; be = (zz+D-xk[k])/s
        cev, cno = cfun(al, be)
        W[k] = mm@(cev+cno)
        P = np.where(al > 0, ndtr(-al)-ndtr(-be), ndtr(be)-ndtr(al))     # exact P_k(z_i), the expression of mc83.cfun
        ENP += mm@(1-P)
        kmass[k] = _kernel(z, xk[k], s, dz, gh)
        en, mass[k], minm[k], lo, hi = _ab_step(m, mn, gh, Cg, Cm, M, wb)
        EN += en; m, mn = mn, m
    return dict(EN=EN, ENP=ENP, W=W, mass=mass, minm=minm, kmass=kmass, G=ng, M=M, Delta_over_h=D/dz)

def _warmup():
    """compile the numba kernels once (JIT excluded from the steady-state runtime)."""
    z = np.linspace(-1.0, 1.0, 101); gh = np.empty(101); m = np.full(101, 1/101); mn = np.empty(101)
    Cg = np.empty(102); Cm = np.empty(102)
    _kernel(z, 0.0, 0.1, z[1]-z[0], gh); _ab_step(m, mn, gh, Cg, Cm, 10, 0.5); _ab_step(m, mn, gh, Cg, Cm, 10, 0.0)


# --- shared: one evaluation and its record (spec §6) ------------------------------------------------------
def evaluate(version, name, n, margin_mult=8):
    """one evaluation; returns the record of spec §6 (without the U-relative quantities, added by relative())."""
    xk, Hk = input_arrays(name); margin = margin_mult*D
    t0 = time.perf_counter()
    r = run_v0(xk, n, margin) if version == 'v0' else run_ab(xk, n, margin, version)
    J = (Hk*r['W'][:, None]).T@Hk/s**2
    runtime = time.perf_counter()-t0
    sgn, ld = np.linalg.slogdet(J)
    Jfull = Hk.T@Hk/s**2
    rec = dict(version=version, input=name, n=n, margin=f'{margin_mult}Delta', G=r['G'], M=r['M'], Delta_over_h=r['Delta_over_h'],
               EN_Phat=(r['EN'] if version != 'v0' else None), EN_P=r['ENP'], S=float(r['W'].sum()),
               logdet=float(ld), logdet_sign=float(sgn), runtime_s=runtime,
               mass_dev_max=float(np.abs(r['mass']-1).max()), mass_last=float(r['mass'][-1]), min_m=float(r['minm'].min()),
               kernel_mass_dev_max=float(np.abs(r['kmass']-1).max()),
               kernel_factor_dev_max=(float(np.abs(1/r['kmass']-1).max()) if version != 'v0' else None),
               w_min=float(r['W'].min()), w_max=float(r['W'].max()),
               lam_min_J=float(np.linalg.eigvalsh(J)[0]), lam_min_Jfull_minus_J=float(np.linalg.eigvalsh(Jfull-J)[0]),
               lam_max_Jfull=float(np.linalg.eigvalsh(Jfull)[-1]))
    rec['EN'] = rec['EN_Phat'] if version != 'v0' else rec['EN_P']           # main E[N] (config conventions.EN_main)
    rec['inv_S'] = bool(1 <= rec['S'] <= K)
    rec['inv_J'] = bool(rec['lam_min_J'] >= 0 and rec['lam_min_Jfull_minus_J'] >= 0)
    rec['inv_w'] = bool(rec['w_min'] >= 0 and rec['w_max'] <= 1)
    return rec, J, r

def c_invariants(version, name, n, margin_mult=8):
    """0 <= c_k(z_i) <= 1 on every node of the grid at every step k = 2..K, with the version's own c formula
    (v0: the prototype's inline formula; A/B: mc83.cfun, hence identical for A and B). Tail = |alpha| > 30 and |beta| > 30.
    Untimed; separate from the evaluation because it covers all nodes, not only those where m_k != 0."""
    xk, _ = input_arrays(name); z, _ = make_grid(xk, n, margin_mult*D, version)
    cmin = np.inf; cmax = -np.inf; nnan = 0; tmin = np.inf; tmax = -np.inf; tnan = 0; ntail = 0
    ph = lambda u: np.exp(-0.5*u*u)/np.sqrt(2*np.pi)
    for k in range(1, K):
        al = (z-D-xk[k])/s; be = (z+D-xk[k])/s
        if version == 'v0':
            P = ndtr(be)-ndtr(al)
            c = 1-P-al*ph(al)+be*ph(be) + np.where(P > 1e-300, (ph(al)-ph(be))**2/np.maximum(P, 1e-300), 0.0)
        else:
            cev, cno = cfun(al, be); c = cev+cno
        bad = np.isnan(c); nnan += int(bad.sum()); cmin = min(cmin, np.nanmin(c)); cmax = max(cmax, np.nanmax(c))
        tl = (np.abs(al) > TAIL) & (np.abs(be) > TAIL); ntail += int(tl.sum())
        if tl.any():
            ct = c[tl]; tnan += int(np.isnan(ct).sum()); tmin = min(tmin, np.nanmin(ct)); tmax = max(tmax, np.nanmax(ct))
    return dict(c_min=float(cmin), c_max=float(cmax), c_nan=nnan, c_nodes=int(len(z)*(K-1)),
                tail_nodes=ntail, tail_c_min=float(tmin), tail_c_max=float(tmax), tail_nan=tnan,
                inv_c=bool(nnan == 0 and cmin >= 0 and cmax <= 1))

def relative(rec, J, JU):
    """U-relative quantities at the same version, grid and margin: Delta m_D, nine lambda_i (J v = lambda J_U v), m_E."""
    lam = eigh(J, JU, eigvals_only=True)
    rec.update(dmD=float(rec['logdet']-np.linalg.slogdet(JU)[1]), lam=[float(v) for v in lam], mE=float(lam[0]))
    return rec

def tag(rec): return f"{rec['version']}/{rec['input']}/n={rec['n']}/{rec['margin']}"

LOG_KEYS = ['G', 'M', 'Delta_over_h', 'EN', 'EN_Phat', 'EN_P', 'S', 'logdet', 'logdet_sign', 'dmD', 'mE',
            'mass_dev_max', 'mass_last', 'min_m', 'kernel_mass_dev_max', 'kernel_factor_dev_max', 'w_min', 'w_max',
            'lam_min_J', 'lam_min_Jfull_minus_J', 'lam_max_Jfull', 'inv_S', 'inv_J', 'inv_w', 'runtime_s']

def log_record(rec, out=print):
    """one log line per number: EV <version>/<input>/n=<n>/<margin> <key> <value>"""
    for k in LOG_KEYS:
        if k not in rec: continue
        v = rec[k]
        out(f"EV {tag(rec)} {k} {'not_applicable' if v is None else repr(v)}")
    for i, v in enumerate(rec.get('lam', [])): out(f"EV {tag(rec)} lam{i+1} {v!r}")

def save(rec, J, r, suffix=''):
    os.makedirs(RES, exist_ok=True)
    b = f"{rec['version']}_{rec['input']}_n{rec['n']}_m{rec['margin'][:-5]}{suffix}"
    np.save(os.path.join(RES, f'J_{b}.npy'), J); np.save(os.path.join(RES, f'w_{b}.npy'), r['W'])
    np.save(os.path.join(RES, f'mass_{b}.npy'), r['mass'])
    return b


# --- unit tests of step 2 -----------------------------------------------------------------------------------
def single_step(version, h, Dl=20.0, r=-20.0, L=80.0):
    """Phat at the reference node r for a standard-normal measurement (x = 0, sigma = 1), threshold Dl, nodes z_i = -L + i h
    formed as j h, j = -L/h .. L/h (one rounding per node; np.arange accumulates a node-position error of ~1e-11 here)."""
    N = int(round(L/h)); z = np.arange(-N, N+1)*h; i = int(round((r+L)/h)); M = int(round(Dl/h))
    assert abs(z[i]-r) < 1e-9 and abs(Dl/h-M) < 1e-9
    gh = np.empty(z.size); C = np.empty(z.size+1)
    _kernel(z, 0.0, 1.0, h, gh); _prefix(gh, C)
    return _window(gh, C, i, M, WB[version])[0]

def unit_tests():
    sys.stdout = Tee(os.path.join(LOGS, 'unit_tests.log'))
    print(f"=== density_recursion_v1.py unit tests (step 2); config {CFG['spec']}; {CFG['hardware']}")
    print("run 2. Run 1 (logs/unit_tests_run1_arange_grid.log) built the A/B and test-(b) nodes with np.arange; (b) then missed "
          "0.483377404983273 by -1.8e-12 through arange's accumulated node-position error. A/B now use z_0 + i h (spec §4).")
    _warmup(); ok = True
    # (a) v0, n = 12, U reproduces the prototype: printed precision (spec §3) and bitwise against density_recursion.py run()
    rec, J, r = evaluate('v0', 'U', 12)
    pa = f"{rec['EN']:.2f}" == '316.95' and f"{rec['logdet']:.4f}" == '197.2023'
    print(f"(a) v0/U/n=12: E[N] = {rec['EN']:.4f} (printed {rec['EN']:.2f}, spec 316.95), log det = {rec['logdet']:.5f} "
          f"(printed {rec['logdet']:.4f}, spec 197.2023), S = {rec['S']:.3f} (spec §3 466.21) -> {'PASS' if pa else 'FAIL'}")
    os.chdir(PKG); ns = {}
    exec(open('density_recursion.py').read().split('\nfor npr in')[0], ns)   # the prototype's setup and run(), not its driver
    os.chdir(_cwd)
    EN0, W0, J0, mass0, G0, ratio0 = ns['run'](12)
    same = (EN0 == rec['EN'] and np.array_equal(W0, r['W']) and np.array_equal(J0, J) and G0 == rec['G']
            and np.array_equal(np.asarray(mass0), r['mass'][1000:K:1000]))
    print(f"(a) bitwise vs prototype run(12): E[N] {EN0!r} vs {rec['EN']!r}; W, J, G, mass every 1000 steps identical: {same} "
          f"-> {'PASS' if same else 'FAIL'}")
    ok &= pa and same
    # (b) single step, standard-normal measurement, Delta = 20, r = -20; true value 0.5
    vA = single_step('A', 1/12); vB = single_step('B', 1/12); vA2 = single_step('A', 1/24); vB2 = single_step('B', 1/24)
    eA, eA2 = 0.5-vA, 0.5-vA2
    pb = abs(vA-0.483377404983273) < 5e-16 and abs(vB-0.5) <= 1e-6 and abs(eA/eA2-2) < 0.01
    print(f"(b) h=1/12: A = {vA:.15f} (spec 0.483377404983273, diff {vA-0.483377404983273:+.1e}, criterion < 5e-16 = printed precision); "
          f"B = {vB:.15f} (|B-0.5| = {abs(vB-0.5):.1e} <= 1e-6: {abs(vB-0.5) <= 1e-6})")
    print(f"(b) h=1/24: A = {vA2:.15f}, B = {vB2:.15f} (|B-0.5| = {abs(vB2-0.5):.1e}); A error {eA:.6e} -> {eA2:.6e}, "
          f"ratio {eA/eA2:.6f} (halving h halves A's error; criterion |ratio - 2| < 0.01) -> {'PASS' if pb else 'FAIL'}")
    ok &= pb
    # (c) conservation and non-negativity, A and B, 5000 steps, input U, all three grids
    for v in ('A', 'B'):
        for n in CFG['grids']['n']:
            rec, J, r = evaluate(v, 'U', n)
            pc = rec['mass_dev_max'] <= 1e-10 and rec['min_m'] >= 0
            print(f"(c) {v}/U/n={n}: max_k |sum m - 1| = {rec['mass_dev_max']:.3e} (<= 1e-10), min m = {rec['min_m']:.3e} (>= 0), "
                  f"G = {rec['G']}, E[N] = {rec['EN']:.4f}, log det = {rec['logdet']:.5f}, runtime {rec['runtime_s']:.2f} s "
                  f"-> {'PASS' if pc else 'FAIL'}")
            ok &= pc
    print(f"unit tests: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', choices=VERSIONS); ap.add_argument('--input', choices=list(CFG['inputs']))
    ap.add_argument('--n', type=int, default=12); ap.add_argument('--margin', type=int, default=8)
    ap.add_argument('--test', action='store_true')
    A_ = ap.parse_args()
    if A_.test: sys.exit(0 if unit_tests() else 1)
    _warmup()
    rec, J, r = evaluate(A_.version, A_.input, A_.n, A_.margin)
    if A_.input != 'U': relative(rec, J, evaluate(A_.version, 'U', A_.n, A_.margin)[1])
    log_record(rec)
