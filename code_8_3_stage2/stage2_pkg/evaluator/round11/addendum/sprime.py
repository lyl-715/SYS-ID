"""Round 11b (addendum): the SoD classes S, S' and S u S' on the full sigma/24 design grid - edge action sets made of product blocks
(each block: an edge list E and a minimum distance MIN from r), the Bellman step over them, tables, and a resumable price scan.

Classes (DECISION 11b-1 and the user's choice of 2026-11-07, DECISION 11b-2):
  S       round 11: block (E, 2 sigma), E = {1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4, inf}, 100 actions;
  S'      DECISION 11b-1: block (E', 1 sigma), E' = {0.5, 0.75} u {1.0, 1.1, ..., 3.0} u {3.5, 4, 5, 6, inf}, 784 actions;
  S u S'  the per-node union of the two blocks (the S block first, then S'), 884 actions. S' alone does not contain S (1.25, 1.75,
          2.25 are not in E', and the r-side clamp r -/+ 2 sigma of S is not an S' edge), so the scan runs on the union, which contains
          both classes exactly; T_1 is a table of the union with its round-11 codes unchanged (codes 0..99 = the S block).

Round 11's dp11.py and eval11.py are reused, not changed:
  * eval11.forward / evaluate / backward / evaluate_refined take any object with rows_full(p) -> (L, U) and are used as they are
    (forward, fixed-policy backward, sigma/120 entry, conservation / non-negativity / zero boundary mass assertions);
  * dp11.Design11 supplies the per-step arrays (u, ndtr(u), ndtr(-u), phi(u) on the frozen cache range, kernels, g_1), the SoD
    reference and gamma_from_J; DesignG subclasses it and adds only dp_g;
  * dp11.Scan11 supplies the frozen price-scan logic (scan(): lambda_0 = 5 gbar, doubling, x0.8, 8 bisections, final = largest F
    among sigma/24-feasible runs, ties -> smaller lambda, N(lambda) monotonicity); ScanG subclasses it and overrides run() only
    (generic DP and table, checkpoint archive for resuming, incremental deletion of non-final run tables).
dp11._dp_step and eval11._rows hard-code a 2-sigma minimum (48 grid units), integer edge offsets 24 e and a single product block.
The kernel _dp_step_u below is dp11._dp_step with exactly three changes: (i) half-widths from per-step edge nodes,
L = am (i - LN_k[j]) + bmin[j], U = am (UN_k[j] - i) + bmin[j]; (ii) the clamp max(., MIN_j) with MIN_j the minimum of the action's
block; (iii) a loop over blocks, code = off_b + (j_lo - lo_b) n_b + (j_hi - lo_b). Every floating-point expression of the Bellman
step is unchanged and in the same order, so with the single S block it reproduces dp11 bit for bit (check 1).

Edge nodes (DECISION 11b-3). u_k = (xbar_k - z_0)/h (eval11.UK). For finite e: lower x-bar edge node LN_k(e) = floor(u_k - 24 e),
upper UN_k(e) = ceil(u_k + 24 e) (outward rounding, "as before"); 24 e is formed exactly as a rational and converted to the nearest
double. For integer 24 e these equal fl_k - 24 e and cl_k + 24 e of round 11 exactly (asserted). For non-integer 24 e the distance of
u_k -/+ 24 e from the nearest integer is asserted > 1e-9 at every step (the floating-point subtraction cannot move the node).
Window of an action of block (E, MIN) at node i: lower edge node min(LN_k(e-), i - MIN), upper max(UN_k(e+), i + MIN), i.e.
L = max(i - LN_k(e-), MIN), U = max(UN_k(e+) - i, MIN); e = inf: L = MIN (U = MIN). Ties -> the first code (strict >).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__)); R11 = os.path.dirname(HERE); EV = os.path.dirname(R11)
sys.path.insert(0, R11)
import numpy as np
from numba import njit, prange
import eval11 as e11                       # round 11, read-only
import dp11                                # round 11, read-only
ew, dr = e11.ew, e11.dr
s, K, B, G = e11.s, e11.K, e11.B, e11.G
Z, DZ, XK, HK, KA, KB, FL, CL, Z0, UK = e11.Z, e11.DZ, e11.XK, e11.HK, e11.KA, e11.KB, e11.FL, e11.CL, e11.Z0, e11.UK
X = e11.X_PAD

E_S = ('1', '1.25', '1.5', '1.75', '2', '2.25', '2.5', '3', '4', 'inf')                                  # round 11 (e11.E_SIGMA)
E_SP = ('0.5', '0.75') + tuple(f"{v/10:.1f}" for v in range(10, 31)) + ('3.5', '4', '5', '6', 'inf')    # DECISION 11b-1
assert len(E_SP) == 28 and tuple(float(v) for v in E_S) == tuple(float(v) for v in e11.E_SIGMA)
MIN_S, MIN_SP = 48, 24                                                                                    # 2 sigma, 1 sigma


class ActionSetU:
    """union of product blocks [(E_b, MIN_b), ...]; the action lists of a block are the same on both sides. Per-action arrays over the
    concatenated lists: am (1 finite, 0 inf), bmin (0 finite, MIN_b inf), minv (MIN_b); per-step edge nodes LN[p, j], UN[p, j] (int64,
    array index p = 0..K-1, row 0 unused); block ranges blo, bhi and code offsets boff."""
    def __init__(self, name, blocks):
        self.name = name; self.blocks = [(tuple(E), int(m)) for E, m in blocks]
        am, bmin, minv, n24, labs = [], [], [], [], []
        blo, bhi, boff = [], [], []; off = 0
        for E, m in self.blocks:
            blo.append(len(am)); boff.append(off)
            for e in E:
                f = e != 'inf'
                am.append(1 if f else 0); bmin.append(0 if f else m); minv.append(m)
                n24.append(Fraction(24)*Fraction(e) if f else None); labs.append(f"{e}@{m // 24}s")
            bhi.append(len(am)); off += len(E)**2
        self.n = off; assert self.n <= 0xFFFF
        self.am = np.array(am, np.int64); self.bmin = np.array(bmin, np.int64); self.minv = np.array(minv, np.int64)
        self.blo = np.array(blo, np.int64); self.bhi = np.array(bhi, np.int64); self.boff = np.array(boff, np.int64)
        self.nblk = len(self.blocks); self.n24 = n24; self.side_labels = labs
        nj = len(am)
        self.LN = np.zeros((K, nj), np.int64); self.UN = np.zeros((K, nj), np.int64)
        self.min_frac_dist = np.inf
        for j, q in enumerate(n24):
            if q is None: continue
            qf = float(q); lo = UK-qf; hi = UK+qf
            self.LN[:, j] = np.floor(lo).astype(np.int64); self.UN[:, j] = np.ceil(hi).astype(np.int64)
            if q.denominator == 1:
                nq = int(q)
                assert np.array_equal(self.LN[:, j], FL-nq) and np.array_equal(self.UN[:, j], CL+nq), (j, q)
            else:
                d = float(min(np.abs(lo-np.round(lo)).min(), np.abs(hi-np.round(hi)).min()))
                self.min_frac_dist = min(self.min_frac_dist, d)
                assert d > 1e-9, (j, q, d)

    def edge_check(self):
        """outward rounding in physical units: z(LN) <= xbar - e sigma < z(LN) + h and z(UN) - h < xbar + e sigma <= z(UN), within
        1e-6 grid steps (as eval11.check_edges); returns the number of violations and the largest offsets (grid units)."""
        viol = 0; wl = 0.0; wh = 0.0
        for j, q in enumerate(self.n24):
            if q is None: continue
            ev = float(q/24)
            dl = ((XK-ev*s)-(Z0+self.LN[:, j]*DZ))/DZ; du = ((Z0+self.UN[:, j]*DZ)-(XK+ev*s))/DZ
            viol += int(((dl < -1e-6) | (dl >= 1+1e-6)).sum()+((du < -1e-6) | (du >= 1+1e-6)).sum())
            wl = max(wl, float(dl.max())); wh = max(wh, float(du.max()))
        return dict(violations=viol, max_lower_offset_grid=wl, max_upper_offset_grid=wh,
                    min_frac_dist_nonint=(None if not np.isfinite(self.min_frac_dist) else self.min_frac_dist), pass_=bool(viol == 0))

    def describe(self):
        return dict(name=self.name, blocks=[dict(E=list(E), MIN_grid=m, MIN_sigma=m/24, n_actions=len(E)**2) for E, m in self.blocks],
                    n_actions=self.n, code='off_b + (j_lo - lo_b) n_b + (j_hi - lo_b), blocks in the listed order')

def set_S(): return ActionSetU('S', [(E_S, MIN_S)])
def set_Sprime(): return ActionSetU("S'", [(E_SP, MIN_SP)])
def set_union(): return ActionSetU("S u S'", [(E_S, MIN_S), (E_SP, MIN_SP)])


@njit(cache=True)
def _decode(c, nblk, blo, bhi, boff):
    b = 0
    for t in range(1, nblk):
        if c >= boff[t]: b = t
    nb = bhi[b]-blo[b]; r = c-boff[b]
    return blo[b]+r//nb, blo[b]+r % nb

@njit(cache=True)
def _rows_u(code, nblk, blo, bhi, boff, am, ln, un, bmin, minv, Lout, Uout):
    """half-widths (grid units) of every node of the full grid for one step's code row."""
    for i in range(code.size):
        im, ip = _decode(np.int64(code[i]), nblk, blo, bhi, boff)
        Lm = am[im]*(i-ln[im])+bmin[im]
        if Lm < minv[im]: Lm = minv[im]
        Up = am[ip]*(un[ip]-i)+bmin[ip]
        if Up < minv[ip]: Up = minv[ip]
        Lout[i] = Lm; Uout[i] = Up

class TableG:
    """codes[(K-1), G] uint16 (row r <-> step k = r + 2, array index p = r + 1) on the full sigma/24 grid, union action set."""
    def __init__(self, codes, acts, tag=''):
        assert codes.shape == (K-1, G) and codes.dtype == np.uint16
        self.codes, self.acts, self.tag = codes, acts, tag
        self.X = X
        self._L = np.empty(G, np.int64); self._U = np.empty(G, np.int64)

    def rows_full(self, p, out=None):
        L, U = (self._L, self._U) if out is None else out
        a = self.acts
        _rows_u(self.codes[p-1], a.nblk, a.blo, a.bhi, a.boff, a.am, a.LN[p], a.UN[p], a.bmin, a.minv, L, U)
        return L, U

    def rows(self, p, a_, b_):
        L, U = self.rows_full(p)
        return L[a_:b_+1].copy(), U[a_:b_+1].copy()

    def is_sod(self): return False


@njit(parallel=True, cache=True)
def _dp_step_u(ng, X, nblk, blo, bhi, boff, am, ln, un, bmin, minv, gE, LgE, SgE, gvE, LgvE, SgvE, VnE, uE, PP, PN, PH, gam, lam, Vout, act):
    """dp11._dp_step over a union of product blocks with generic half-widths; every floating-point expression is dp11's, in the
    same order."""
    L = ng+2*X
    nbad = 0; nout = 0
    for o in prange(ng):
        ei = o+X; Vi = VnE[ei]
        best = -np.inf; bc = 0
        for b in range(nblk):
            lo_b = blo[b]; nb = bhi[b]-lo_b; off = boff[b]
            for im in range(lo_b, bhi[b]):
                Lm = am[im]*(o-ln[im])+bmin[im]
                if Lm < minv[im]: Lm = minv[im]
                el = ei-Lm
                if el < 0:
                    nout += 1; el = 0
                al = uE[el]; pa = PH[el]; gl = gE[el]; gvl = gvE[el]; Ll = LgE[el]; Ll1 = LgE[el+1]; Lvl = LgvE[el]
                PPl = PP[el]; PNl = PN[el]; pos = al > 0
                for ip in range(lo_b, bhi[b]):
                    Up = am[ip]*(un[ip]-o)+bmin[ip]
                    if Up < minv[ip]: Up = minv[ip]
                    eu = ei+Up
                    if eu > L-1:
                        nout += 1; eu = L-1
                    be = uE[eu]; pb = PH[eu]
                    if pos: P = PNl-PN[eu]
                    else: P = PP[eu]-PPl
                    cev = 1-P-al*pa+be*pb
                    if P > 1e-300:
                        d = pa-pb; cno = d*d/P
                    else:
                        cno = 0.0
                    half = 0.5*(gl+gE[eu])
                    q = (Ll+SgE[eu])+half
                    Ph = (LgE[eu]-Ll1)+half
                    snd = (Lvl+SgvE[eu])+0.5*(gvl+gvE[eu])
                    v = gam*(cev+cno)-lam*q+Ph*Vi+snd
                    if v > best:
                        best = v; bc = off+(im-lo_b)*nb+(ip-lo_b)
        if not (best > -np.inf and best < np.inf): nbad += 1
        Vout[o] = best; act[o] = bc
    return nbad, nout


def warmup():
    dp11.warmup()
    a = set_S(); ng = 8; Xw = 4; L = ng+2*Xw
    ln = np.array([1]*a.am.size, np.int64)
    _dp_step_u(ng, Xw, a.nblk, a.blo, a.bhi, a.boff, a.am, ln, ln, a.bmin, a.minv, *[np.zeros(L) for _ in range(11)], 1.0, 1.0,
               np.empty(ng), np.empty(ng, np.uint16))
    _rows_u(np.zeros(ng, np.uint16), a.nblk, a.blo, a.bhi, a.boff, a.am, ln, ln, a.bmin, a.minv, np.empty(ng, np.int64), np.empty(ng, np.int64))


class DesignG(dp11.Design11):
    """dp11.Design11 (per-step arrays, kernels, SoD reference, gamma_from_J) plus the generic Bellman recursion."""
    def dp_g(self, acts, gam, lam, codes=None):
        """backward induction at price lam over `acts`; argmax codes into codes[(K-1), G] if given; returns (V_lambda =
        sum_i ghat_{1,i} V_2(i), number of non-finite values). Same structure as dp11.Design11.dp."""
        bufs = self.ext_buffers(); uE, PP, PN, PH, gE, LgE, SgE, VnE, gvE, LgvE, SgvE = bufs
        Vn = np.zeros(G); Vout = np.empty(G); act = np.empty(G, np.uint16); nbad = 0; nout = 0
        for p in range(K-1, 0, -1):
            self.prep(p, bufs)
            dp11._value_ext_full(Vn, X, gE, VnE, gvE, LgvE, SgvE)
            nb, no = _dp_step_u(G, X, acts.nblk, acts.blo, acts.bhi, acts.boff, acts.am, acts.LN[p], acts.UN[p], acts.bmin, acts.minv,
                                gE, LgE, SgE, gvE, LgvE, SgvE, VnE, uE, PP, PN, PH, float(gam[p]), float(lam), Vout, act)
            nbad += nb; nout += no
            if codes is not None: codes[p-1, :] = act
            Vn, Vout = Vout, Vn
        if nout: raise e11.AssertionStop(f"DP: {nout} window edges outside the padded index range")
        return float(self.g1[self.a1:self.b1+1]@Vn[self.a1:self.b1+1]), nbad


class ScanG(dp11.Scan11):
    """dp11.Scan11 with run() overridden: generic DP and table; every finished run is appended (fsync) to a JSONL archive and
    replayed on a restart (same lambda, same run number: the lambda sequence is deterministic); only the table of the current
    best feasible run (largest F, ties -> smaller lambda, as dp11.select_final) is kept on disk, every other run table is deleted as
    soon as its record is written."""
    def __init__(self, des, acts, gam, gbar, g1, label, work, log, archive, max_runs=dp11.MAX_RUNS):
        super().__init__(des, acts, gam, gbar, g1, label, work, log)
        self.state['actions'] = acts.name
        self.archive, self.max_runs = archive, int(max_runs)
        self.saved = {}
        if os.path.exists(archive):
            for l in open(archive):
                r = json.loads(l); self.saved[r['run']] = r
        self.best = None                         # (F, lam, table path) of the best feasible run so far
        self.n_resumed = 0

    def _better(self, r):
        return r['feasible'] and (self.best is None or r['F'] > self.best[0] or (r['F'] == self.best[0] and r['lam'] < self.best[1]))

    def run(self, lam, phase):
        if lam in self.memo: return self.memo[lam]
        n = len(self.state['runs'])
        if n >= self.max_runs: raise dp11.RunCap()
        j = n+1
        if j in self.saved:
            r = self.saved[j]
            assert r['lam'] == lam and r['phase'] == phase, (j, r['lam'], lam, r['phase'], phase)
            if self._better(r):
                assert os.path.exists(r['table']), f"resume: kept table of run {j} missing ({r['table']})"
                self.best = (r['F'], r['lam'], r['table'])
            self.n_resumed += 1
            self.log(f"LAM {self.label} run={j:02d} RESUMED from {os.path.basename(self.archive)} lam={lam!r} EN={r['EN']!r} F={r['F']!r} "
                     f"V={r['dsum']!r} dual={r['dual']!r}")
        else:
            tpath = os.path.join(self.work, f"{self.label}_r{j:02d}.npy")
            codes = np.lib.format.open_memmap(tpath, mode='w+', dtype=np.uint16, shape=(K-1, G))
            t0 = time.perf_counter(); dsum, nbad = self.des.dp_g(self.acts, self.gam, lam, codes); t_dp = time.perf_counter()-t0
            codes.flush(); del codes
            codes = np.load(tpath, mmap_mode='r'); tab = TableG(codes, self.acts, f"{self.label}_r{j:02d}")
            t0 = time.perf_counter(); rec, J, fw = e11.evaluate(tab, gam=self.gam); t_fw = time.perf_counter()-t0
            W = fw['W']; F = float(self.gam[1:]@W[1:]); EN = rec['EN']; N = EN-1.0
            dual = self.g1+dsum+lam*(B-1.0)
            r = dict(run=j, phase=phase, lam=lam, lam_over_gbar=lam/self.gbar, dual=dual, EN=EN, N=N, S=rec['S'], F=F,
                     feasible=bool(EN <= B), slack=B-EN, logdet=rec['logdet'], dmD=rec['logdet']-self.ld_sod,
                     mass_dev_max=rec['mass_dev_max'], min_m=rec['min_m'], nonfinite_V=int(nbad), dp_s=t_dp, fwd_s=t_fw, table=tpath,
                     dsum=dsum, dp_minus_policy=dsum-(F-lam*N), policy_minus_sod=(F-lam*N)-(self.F_sod-lam*self.N_sod),
                     support_lo_min=int(fw['sup_lo'].min()), support_hi_max=int(fw['sup_hi'].max()))
            r['dp_minus_policy_rel'] = abs(r['dp_minus_policy'])/abs(F-lam*N)
            del codes, tab
            keep = self._better(r); prev = self.best[2] if (keep and self.best is not None) else None
            r['kept'] = bool(keep)
            with open(self.archive, 'a') as f:
                f.write(json.dumps(r)+'\n'); f.flush(); os.fsync(f.fileno())
            self.saved[j] = r
            if keep:
                self.best = (r['F'], r['lam'], tpath)
                if prev and prev != tpath and os.path.exists(prev): os.remove(prev)
            elif os.path.exists(tpath):
                os.remove(tpath)
            self.log(f"LAM {self.label} run={j:02d} phase={phase} lam={lam!r} lam/gbar={lam/self.gbar!r} dual={dual!r} EN={EN!r} "
                     f"S={r['S']!r} F={F!r} V={dsum!r} feasible={r['feasible']} slack={r['slack']!r} logdet={r['logdet']!r} "
                     f"dp_s={t_dp:.1f} fwd_s={t_fw:.1f} [diag: V-policy rel {r['dp_minus_policy_rel']!r}, mass dev {r['mass_dev_max']!r}, "
                     f"support [{r['support_lo_min']}, {r['support_hi_max']}]; table {'kept' if keep else 'deleted'}]")
        self.state['runs'].append(r); self.memo[lam] = r
        mv = dp11.mono_check(self.state['runs']); self.state['mono_worst'] = mv[0]; self.state['mono_pair'] = mv[1]
        if mv[0] > dp11.MONO_TOL: raise dp11.MonoStop(mv)
        return r

    def cleanup(self, keep):
        for r in self.state['runs']:
            if r['table'] not in keep and os.path.exists(r['table']): os.remove(r['table'])
