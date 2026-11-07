"""Round 11: stand-alone DP on the FULL design grid with a generic product action set, the frozen price scan of Section IV-D and
the iteration of the tangent point A (spec reports_CN/plan_round11_edge_actions_EN.md §2-§3, DECISIONS 11-1..11-3; execution
prompt reports_CN/prompt_round11_claude_code_EN.md §2.1-§2.4).

Bellman step (copied from round4/trigger_dp.py: _value_ext, _dp_step), reward gamma_k c_{k,i} - lambda qhat_{k,i}:
  V_k(i) = max_a { gamma_k c + (-lambda) qhat + Phat V_{k+1}(i) + [left tail + right tail of ghat V_{k+1}] },  V_{K+1} = 0,
k = K..2 (step K's action optimised; step 1 is the forced send). qhat, Phat, the send-branch sum and c are formed exactly as in the
frozen DP: tails of ghat (one prefix and one suffix pass), c from the per-step node arrays u = (r - x_k)/sigma, ndtr(u), ndtr(-u),
phi(u) (scipy / numpy, the functions cfun uses), alpha = u_{i - L}, beta = u_{i + U}; argmax stored as uint16 per (k, node), ties ->
the first action in code order (strict >, codes in increasing order).

Changed (DECISION 11-3): the state support is the whole design grid at every step (V_{k+1} is the computed value at every node, never
taken as 0 outside a moving window); edges may lie anywhere in [-X, G-1+X]. The per-step arrays u, ndtr(u), ndtr(-u), phi(u) are
taken from scipy/numpy on the frozen cache range [a_k - 960, b_k + 960] (|u| <= 128, the range the round-4 DP cached) and set to
their exact limits outside it (ndtr = 0 or 1 exactly, phi = 0 exactly for |u| > 127; asserted against scipy at the first node
outside the range of every step, and bitwise against full scipy arrays in the self-test). The kernel of step k is the frozen
dr._kernel on the 88-sigma window [a_k, b_k], exactly as the round-4 DP and evaluator compute it.

Price scan (frozen, round4/trigger_dp.Scan.scan): lambda in units of gbar = mean_{k>=2} gamma_k; lambda_0 = 5 gbar; infeasible ->
double (a doubled value above 100 gbar is replaced by 100 gbar, evaluated before stopping); from the feasible lambda multiply by 0.8
until the first infeasible lambda (a lambda < 1e-3 gbar ends the scan); 8 bisections (arithmetic midpoint) keeping the feasible side;
FINAL = largest F = sum_{k>=2} gamma_k w_k among the feasible policies (E[N] <= B on sigma/24), ties -> smaller lambda; N(lambda)
monotonicity checked after every run (relative 1e-9; a violation stops the scan); at most 40 DP runs.

Tangent point: gamma_k = sigma^-2 h_k^T A h_k, A = J_ref^-1 by Cholesky solve (no explicit inverse) exactly as
round4/trigger_dp.Design.gamma; A_0 = J_SoD^-1 (J_SoD from the accepted operator evaluator_fast via evaluator_window.evaluate_table),
A_j = J(T_{j-1})^-1 (J from eval11 on sigma/24).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from numba import njit, prange
import eval11 as e11
ew, dr = e11.ew, e11.dr
s, K, B, G = e11.s, e11.K, e11.B, e11.G
Z, DZ, XK, HK, KA, KB, FL, CL, Z0 = e11.Z, e11.DZ, e11.XK, e11.HK, e11.KA, e11.KB, e11.FL, e11.CL, e11.Z0
X = e11.X_PAD
XC = ew.X24                                    # 960: the frozen round-4 cache extension around the 88-sigma window
MAX_RUNS = 40; N_BISECT = 8; MONO_TOL = 1e-9; LAM_CAP = 100.0


@njit(cache=True)
def _prep(uc, ppc, pnc, phc, c0, gk, ka, kb, z, z0, dz, xk, sg, X, uE, PP, PN, PH, gE, LgE, SgE):
    """per-step extended arrays over [-X, G-1+X] (index e = j + X): u, ndtr(u), ndtr(-u), phi(u) from the cached scipy/numpy values on
    the frozen range [c0, c0 + len(uc) - 1], exact limits outside it (returns the number of nodes outside the range with |u| <= 127,
    which must be 0); kernel gk on [ka, kb], zero elsewhere, and its prefix / suffix tails."""
    ng = z.size; L = ng+2*X; nc = uc.size; nbad = 0
    for e in range(L):
        j = e-X
        if j >= c0 and j < c0+nc:
            uE[e] = uc[j-c0]; PP[e] = ppc[j-c0]; PN[e] = pnc[j-c0]; PH[e] = phc[j-c0]
        else:
            zj = z[j] if (j >= 0 and j < ng) else z0+j*dz
            u = (zj-xk)/sg
            uE[e] = u
            if u > 0.0:
                PP[e] = 1.0; PN[e] = 0.0
            else:
                PP[e] = 0.0; PN[e] = 1.0
            PH[e] = 0.0
            if abs(u) <= 127.0: nbad += 1
        gE[e] = gk[j-ka] if (j >= ka and j <= kb) else 0.0
    LgE[0] = 0.0
    for e in range(1, L): LgE[e] = LgE[e-1]+gE[e-1]
    SgE[L-1] = 0.0
    for e in range(L-2, -1, -1): SgE[e] = SgE[e+1]+gE[e+1]
    return nbad

@njit(cache=True)
def _value_ext_full(Vn, X, gE, VnE, gvE, LgvE, SgvE):
    """V_{k+1} on the extended range (the computed value at every grid node, 0 on the padding) and the tails of gv = ghat V_{k+1}
    (prefix and suffix passes, as round4/trigger_dp._value_ext)."""
    ng = Vn.size; L = ng+2*X
    for e in range(L):
        j = e-X
        VnE[e] = Vn[j] if (j >= 0 and j < ng) else 0.0
        gvE[e] = gE[e]*VnE[e]
    LgvE[0] = 0.0
    for e in range(1, L): LgvE[e] = LgvE[e-1]+gvE[e-1]
    SgvE[L-1] = 0.0
    for e in range(L-2, -1, -1): SgvE[e] = SgvE[e+1]+gvE[e+1]

@njit(parallel=True, cache=True)
def _dp_step(ng, X, fl, cl, am, bm, ap, bp, gE, LgE, SgE, gvE, LgvE, SgvE, VnE, uE, PP, PN, PH, gam, lam, Vout, act):
    """Bellman maximisation over the product action set at every grid node o (extended index o + X); arithmetic of
    round4/trigger_dp._dp_step. Returns (non-finite values, edges outside the padded range)."""
    NM = am.size; NP = ap.size; L = ng+2*X
    nbad = 0; nout = 0
    for o in prange(ng):
        ei = o+X; Vi = VnE[ei]
        best = -np.inf; bc = 0
        for im in range(NM):
            Lm = am[im]*(o-fl)+bm[im]
            if Lm < 48: Lm = 48
            el = ei-Lm
            if el < 0:
                nout += 1; el = 0
            al = uE[el]; pa = PH[el]; gl = gE[el]; gvl = gvE[el]; Ll = LgE[el]; Ll1 = LgE[el+1]; Lvl = LgvE[el]
            PPl = PP[el]; PNl = PN[el]; pos = al > 0
            for ip in range(NP):
                Up = ap[ip]*(cl-o)+bp[ip]
                if Up < 48: Up = 48
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
                    best = v; bc = im*NP+ip
        if not (best > -np.inf and best < np.inf): nbad += 1
        Vout[o] = best; act[o] = bc
    return nbad, nout


def warmup():
    e11.warmup()
    ng = 8; Xw = 4; L = ng+2*Xw; z = np.linspace(0.0, 1.0, ng)
    bufs = [np.empty(L) for _ in range(7)]
    _prep(np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), 2, np.zeros(3), 2, 4, z, 0.0, 0.1, 0.5, 1.0, Xw, *bufs)
    _value_ext_full(np.zeros(ng), Xw, bufs[4], np.empty(L), np.empty(L), np.empty(L), np.empty(L))
    a = e11.edge_set()
    _dp_step(ng, Xw, 3, 4, a.am[:1], np.array([1], np.int64), a.ap[:1], np.array([1], np.int64), *[np.zeros(L) for _ in range(11)],
             1.0, 1.0, np.empty(ng), np.empty(ng, np.uint16))


# --- the design: per-step lambda-independent arrays (frozen cache range) --------------------------------------------------
class Design11:
    def __init__(self, log=print):
        t0 = time.perf_counter()
        self.log = log
        w = (KB-KA+1); assert int(w[1:].min()) == int(w[1:].max()); self.W88 = int(w[1])
        self.NC = self.W88+2*XC
        assert int((KA[1:]-XC).min()) >= 0 and int((KB[1:]+XC).max()) <= G-1
        self.UC = np.empty((K, self.NC)); self.PPC = np.empty((K, self.NC)); self.PNC = np.empty((K, self.NC))
        self.PHC = np.empty((K, self.NC)); self.GK = np.zeros((K, self.W88))
        lim_ok = True
        for p in range(1, K):
            A, Bb = int(KA[p]), int(KB[p])
            uE = (Z[A-XC:Bb+XC+1]-XK[p])/s                     # round4/trigger_dp.Design.__init__, unchanged expressions
            self.UC[p] = uE; self.PPC[p] = dr.ndtr(uE); self.PNC[p] = dr.ndtr(-uE); self.PHC[p] = dr.phi(uE)
            dr._kernel(Z[A:Bb+1], XK[p], s, DZ, self.GK[p])
            # exact limits just outside the cache range (first node on each side), scipy/numpy vs the constants used by _prep
            ul = (Z[A-XC-1]-XK[p])/s; uh = (Z[Bb+XC+1]-XK[p])/s
            vals = (dr.ndtr(np.array([ul, uh])), dr.ndtr(-np.array([ul, uh])), dr.phi(np.array([ul, uh])))
            lim_ok &= bool(vals[0][0] == 0.0 and vals[0][1] == 1.0 and vals[1][0] == 1.0 and vals[1][1] == 0.0
                           and vals[2][0] == 0.0 and vals[2][1] == 0.0 and ul < -127 and uh > 127)
        if not lim_ok: raise e11.AssertionStop("scipy/numpy ndtr, phi not at their exact limits outside the cache range")
        self.g1 = np.empty(G); dr._kernel(Z, XK[0], s, DZ, self.g1)
        self.a1, self.b1 = int(KA[1]), int(KB[1])
        self.cache_s = time.perf_counter()-t0
        log(f"DESIGN11 G = {G} (full grid at every step), kernel windows {self.W88} nodes, cache range {self.NC} nodes per step, "
            f"built in {self.cache_s:.1f} s; limits outside the cache range exact: {lim_ok}")
        self.J_sod = None

    def sod_reference(self):
        """the constant-SoD table through the accepted operator (evaluator_fast via evaluator_window.evaluate_table), bitwise."""
        if self.J_sod is None:
            rec, J, r = ew.evaluate_table(e11.NAME, ew.ConstTable(ew.SOD_M, ew.SOD_M))
            self.rec_sod, self.J_sod, self.W_sod = rec, J, r['W']
        return self.rec_sod, self.J_sod, self.W_sod

    @staticmethod
    def gamma_from_J(J):
        """(gamma_k, gbar, gamma_1) for A = J^-1, Cholesky solve as round4/trigger_dp.Design.gamma."""
        Lc = cholesky(J, lower=True)
        Y = solve_triangular(Lc, HK.T, lower=True)
        g = np.sum(Y*Y, axis=0)/s**2
        return g, float(g[1:].mean()), float(g[0])

    def ext_buffers(self):
        L = G+2*X
        return [np.empty(L) for _ in range(11)]

    def prep(self, p, bufs):
        uE, PP, PN, PH, gE, LgE, SgE = bufs[:7]
        nb = _prep(self.UC[p], self.PPC[p], self.PNC[p], self.PHC[p], int(KA[p])-XC, self.GK[p], int(KA[p]), int(KB[p]), Z, Z0, DZ,
                   float(XK[p]), s, X, uE, PP, PN, PH, gE, LgE, SgE)
        if nb: raise e11.AssertionStop(f"step {p+1}: {nb} nodes outside the cache range with |u| <= 127")

    def dp(self, acts, gam, lam, codes):
        """backward induction at price lam on the full grid; argmax codes into codes[(K-1), G]; returns (sum ghat_1 V_2, nbad)."""
        bufs = self.ext_buffers(); uE, PP, PN, PH, gE, LgE, SgE, VnE, gvE, LgvE, SgvE = bufs
        Vn = np.zeros(G); Vout = np.empty(G); act = np.empty(G, np.uint16); nbad = 0; nout = 0
        for p in range(K-1, 0, -1):
            self.prep(p, bufs)
            _value_ext_full(Vn, X, gE, VnE, gvE, LgvE, SgvE)
            nb, no = _dp_step(G, X, int(FL[p]), int(CL[p]), acts.am, acts.bm, acts.ap, acts.bp, gE, LgE, SgE, gvE, LgvE, SgvE, VnE,
                              uE, PP, PN, PH, float(gam[p]), float(lam), Vout, act)
            nbad += nb; nout += no
            codes[p-1, :] = act
            Vn, Vout = Vout, Vn
        if nout: raise e11.AssertionStop(f"DP: {nout} window edges outside the padded index range")
        return float(self.g1[self.a1:self.b1+1]@Vn[self.a1:self.b1+1]), nbad


# --- price scan (frozen logic of round4/trigger_dp.Scan) ---------------------------------------------------------------
class RunCap(Exception): pass
class MonoStop(Exception): pass

def mono_check(runs):
    rs = sorted(runs, key=lambda r: r['lam']); worst = (0.0, None)
    for i in range(len(rs)):
        for j in range(i+1, len(rs)):
            if rs[j]['lam'] > rs[i]['lam']:
                v = (rs[j]['N']-rs[i]['N'])/abs(rs[i]['N'])
                if v > worst[0]: worst = (v, (rs[i]['run'], rs[j]['run']))
    return worst

def select_final(runs):
    feas = [r for r in runs if r['feasible']]
    if not feas: return None
    return sorted(feas, key=lambda r: (-r['F'], r['lam']))[0]

class Scan11:
    """one price scan for (action set, tangent point). Tables of every run are written to `work` as .npy (uint16, K-1 x G);
    the caller deletes the non-final ones."""
    def __init__(self, des, acts, gam, gbar, g1, label, work, log, ref_I=None):
        self.des, self.acts, self.gam, self.gbar, self.g1 = des, acts, gam, gbar, g1
        self.label, self.work, self.log = label, work, log
        os.makedirs(work, exist_ok=True)
        rec_sod, J_sod, W_sod = des.sod_reference()
        self.ld_sod = float(np.linalg.slogdet(J_sod)[1]); self.F_sod = float(gam[1:]@W_sod[1:]); self.N_sod = float(rec_sod['EN'])-1.0
        self.state = dict(label=label, actions=acts.name, gbar=gbar, gamma1=g1, B=B, runs=[], status='running')
        self.memo = {}

    def run(self, lam, phase):
        if lam in self.memo: return self.memo[lam]
        n = len(self.state['runs'])
        if n >= MAX_RUNS: raise RunCap()
        j = n+1; tpath = os.path.join(self.work, f"{self.label}_r{j:02d}.npy")
        codes = np.lib.format.open_memmap(tpath, mode='w+', dtype=np.uint16, shape=(K-1, G))
        t0 = time.perf_counter(); dsum, nbad = self.des.dp(self.acts, self.gam, lam, codes); t_dp = time.perf_counter()-t0
        codes.flush(); del codes
        codes = np.load(tpath, mmap_mode='r'); tab = e11.Table11(codes, self.acts, f"{self.label}_r{j:02d}")
        t0 = time.perf_counter(); rec, J, fw = e11.evaluate(tab, gam=self.gam, window=(self.acts.name == 'old')); t_fw = time.perf_counter()-t0
        W = fw['W']; F = float(self.gam[1:]@W[1:]); EN = rec['EN']; N = EN-1.0
        dual = self.g1+dsum+lam*(B-1.0)
        r = dict(run=j, phase=phase, lam=lam, lam_over_gbar=lam/self.gbar, dual=dual, EN=EN, N=N, S=rec['S'], F=F,
                 feasible=bool(EN <= B), slack=B-EN, logdet=rec['logdet'], dmD=rec['logdet']-self.ld_sod,
                 mass_dev_max=rec['mass_dev_max'], min_m=rec['min_m'], nonfinite_V=int(nbad), dp_s=t_dp, fwd_s=t_fw, table=tpath,
                 dsum=dsum, dp_minus_policy=dsum-(F-lam*N), policy_minus_sod=(F-lam*N)-(self.F_sod-lam*self.N_sod),
                 support_lo_min=int(fw['sup_lo'].min()), support_hi_max=int(fw['sup_hi'].max()))
        r['dp_minus_policy_rel'] = abs(r['dp_minus_policy'])/abs(F-lam*N)
        self.state['runs'].append(r); self.memo[lam] = r
        mv = mono_check(self.state['runs']); self.state['mono_worst'] = mv[0]; self.state['mono_pair'] = mv[1]
        self.log(f"LAM {self.label} run={j:02d} phase={phase} lam={lam!r} lam/gbar={lam/self.gbar!r} dual={dual!r} EN={EN!r} "
                 f"S={r['S']!r} F={F!r} feasible={r['feasible']} slack={r['slack']!r} logdet={r['logdet']!r} dmD={r['dmD']!r} "
                 f"dp_s={t_dp:.1f} fwd_s={t_fw:.1f} [diag: V-policy rel {r['dp_minus_policy_rel']!r}, support [{r['support_lo_min']}, {r['support_hi_max']}]]")
        if mv[0] > MONO_TOL: raise MonoStop(mv)
        return r

    def scan(self):
        g = self.gbar; note = []; lam_f = lam_i = None
        try:
            lam = 5.0*g; r = self.run(lam, 'start'); cap = LAM_CAP*g
            while not r['feasible']:
                if lam >= cap:
                    self.state['status'] = 'no feasible policy in the lambda range'; return self.state
                lam = min(2.0*lam, cap)
                r = self.run(lam, 'double')
            lam_f = lam; lam_i = None
            while True:
                lam = 0.8*lam
                r = self.run(lam, 'scan')
                if not r['feasible']: lam_i = lam; break
                lam_f = lam
                if lam < 1e-3*g:
                    note.append('no infeasible end point in the lambda range'); break
            if lam_i is not None:
                for _ in range(N_BISECT):
                    mid = 0.5*(lam_f+lam_i)
                    r = self.run(mid, 'bisect')
                    if r['feasible']: lam_f = mid
                    else: lam_i = mid
            self.state['status'] = 'done'
        except RunCap:
            note.append('run cap'); self.state['status'] = 'run cap (40 DP runs) reached'
        except MonoStop as e:
            self.state['status'] = 'monotonicity violation'; self.state['message'] = repr(e.args[0])
            self.log(f"STOP {self.label} monotonicity violation {e.args[0]!r}")
            return self.state
        fin = select_final(self.state['runs'])
        self.state['notes'] = note; self.state['final_run'] = fin['run'] if fin else None
        self.state['min_dual'] = min(r['dual'] for r in self.state['runs']); self.state['lam_bracket'] = [lam_f, lam_i]
        if fin:
            self.log(f"FINAL {self.label} run={fin['run']:02d} lam={fin['lam']!r} F={fin['F']!r} EN={fin['EN']!r} slack={fin['slack']!r} "
                     f"logdet={fin['logdet']!r}; DP runs {len(self.state['runs'])}")
        return self.state

    def cleanup(self, keep):
        """delete the run tables except the paths in `keep`."""
        for r in self.state['runs']:
            if r['table'] not in keep and os.path.exists(r['table']): os.remove(r['table'])
