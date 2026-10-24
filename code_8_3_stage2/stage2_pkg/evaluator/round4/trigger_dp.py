"""Round 4, Part B: DP design of a (k, r)-dependent window table on the accepted reference-value grid, frozen lambda scan,
checks B4 (spec reports_CN/plan_round4_CN.md v3.1, frozen; §1.3, §3 items 3-5).

States: the nodes i of step k's support window [a_k, b_k] (evaluator_window.windows88, R = 88 sigma). Actions:
(M^-, M^+) in {48, 96, ..., 960}^2 (400; code = i_- * 20 + i_+, DECISION B-2). Reward gamma_k c_{k,i} - lambda qhat_{k,i}.
  V_k(i) = max_a { gamma_k c + (-lambda) qhat + Phat V_{k+1}(i) + sum_j ghat_{k,j} (1 - omega_{k,ij}) V_{k+1}(j) },  V_{K+1} = 0,
k = K..2 (step K's action is optimised; step 1 is the forced send). qhat and Phat are formed exactly as in the evaluator
(evaluator_window._tails: one prefix pass and one suffix pass of ghat; qhat = left tail + right tail); the send-branch sum is
the left tail + right tail of ghat V_{k+1} (prefix and suffix passes). c is the formula of dr.cfun (mc83.cfun) evaluated
from per-step node lookups u_j = (r_j - x_k)/sigma, ndtr(u_j), ndtr(-u_j), phi(u_j) (scipy/numpy, the functions cfun uses),
with alpha = u_{i - M^-}, beta = u_{i + M^+} (DECISION B-3). argmax stored as uint16 per (k, node) with the step's support
start index; ties -> the first action in code order (DECISION B-4). V_{k+1} = 0 outside W_{k+1} (DECISION A-3).

Objectives: (i) 'S': gamma = 1; (ii) 'W': gamma_k = sigma^-2 h_k^T A h_k, A = J_SoD(u)^-1 (J from Part A on the SoD table,
which dispatches to the accepted operator; Cholesky solve, no explicit inverse), fixed. F = sum_{k>=2} gamma_k w_k from the
forward pass of the stored table (Part A general path).

Lambda selection (spec §1.3, frozen; lambda in units of gbar = mean_{k>=2} gamma_k, gbar = 1 for 'S'): lambda_0 = 5 gbar;
infeasible -> double (values above 100 gbar are not run, DECISION B-5); from the feasible lambda multiply by 0.8 until the
first infeasible lambda (a lambda < 1e-3 gbar is run and, if feasible, ends the scan, DECISION B-6); 8 bisections
(arithmetic midpoint, DECISION B-7) keeping the feasible side; FINAL = largest F among all evaluated feasible policies
(E[N] <= B), ties -> smaller lambda; monotonicity of N = E[N] - 1 in lambda checked after every run (relative 1e-9, a violation
stops the case); at most 40 DP runs per case.

Usage (from this directory):
  python3 trigger_dp.py scan --case U/S [--case U/W ...]   # lambda scan; checkpoint after every run, re-run resumes
  python3 trigger_dp.py final                              # B4 checks on the final tables, packed copies
  python3 trigger_dp.py time --case U/S                    # one DP at lambda_0 with a per-part timing (no checkpoint)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from scipy.linalg import cholesky, solve_triangular
import numba
from numba import njit, prange
import evaluator_window as ew
dr = ew.dr

s, K = ew.s, ew.K
B = ew.B_BUDGET
X = ew.X24; STEP = ew.STEP_M; NA1 = ew.N_ACT1
TABLES = os.path.join(ew.RES, 'tables'); RUNS = os.path.join(ew.RES, 'runs')
os.makedirs(TABLES, exist_ok=True); os.makedirs(RUNS, exist_ok=True)
CASES = ('U/S', 'U/W', 'P5514/S', 'P5514/W')
MAX_RUNS = 40; N_BISECT = 8; MONO_TOL = 1e-9
PAD = np.uint16(0xFFFF)


# --- numba kernels ---------------------------------------------------------------------------------------------------
@njit(cache=True)
def _value_ext(Vn, an, bn, a, nw, gE, VnE, gvE, LgvE, SgvE):
    """V_{k+1} on the extended range of step k (0 outside [an, bn]) and the tails of gv = ghat V_{k+1}, with the same
    prefix/suffix passes as evaluator_window._gen_back."""
    L = nw+2*X; base = a-X
    for e in range(L):
        j = base+e
        VnE[e] = Vn[j-an] if (j >= an and j <= bn) else 0.0
        gvE[e] = gE[e]*VnE[e]
    LgvE[0] = 0.0
    for e in range(1, L): LgvE[e] = LgvE[e-1]+gvE[e-1]
    SgvE[L-1] = 0.0
    for e in range(L-2, -1, -1): SgvE[e] = SgvE[e+1]+gvE[e+1]

@njit(parallel=True, cache=True)
def _dp_step(nw, gE, LgE, SgE, gvE, LgvE, SgvE, VnE, uE, PP, PN, PH, gam, lam, Vout, act):
    """Bellman maximisation over the 400 actions at every node o of the support window (extended index o + X)."""
    nbad = 0
    for o in prange(nw):
        ei = o+X; Vi = VnE[ei]
        best = -np.inf; bc = 0
        for im in range(NA1):
            el = ei-STEP*(im+1)
            al = uE[el]; pa = PH[el]; gl = gE[el]; gvl = gvE[el]; Ll = LgE[el]; Ll1 = LgE[el+1]; Lvl = LgvE[el]
            PPl = PP[el]; PNl = PN[el]; pos = al > 0
            for ip in range(NA1):
                eu = ei+STEP*(ip+1)
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
                    best = v; bc = im*NA1+ip
        if not (best > -np.inf and best < np.inf): nbad += 1
        Vout[o] = best; act[o] = bc
    return nbad


def _warmup():
    ew._warmup()
    nw = 4; L = nw+2*X; z = np.zeros(L)
    _value_ext(np.zeros(nw), 0, nw-1, X, nw, z, np.empty(L), np.empty(L), np.empty(L), np.empty(L))
    _dp_step(nw, *[np.zeros(L) for _ in range(11)], 1.0, 1.0, np.empty(nw), np.empty(nw, np.uint16))


# --- one input: grid, cache of lambda-independent per-step arrays ----------------------------------------------------
class Design:
    def __init__(self, name, log=print):
        self.name = name
        self.xk, self.Hk, self.z, self.dz, self.a, self.b = ew.design_grid(name)
        self.G = len(self.z); self.w = (self.b-self.a+1).astype(np.int64); self.WMAX = int(self.w[1:].max())
        assert int((self.a-X).min()) >= 0 and int((self.b+X).max()) <= self.G-1
        t0 = time.perf_counter()
        self.cache = [None]*K; gh = np.zeros(self.G)
        for p in range(1, K):
            A, Bb = int(self.a[p]), int(self.b[p]); nw = Bb-A+1; L = nw+2*X
            gh[self.a[p-1]:self.b[p-1]+1] = 0.0
            dr._kernel(self.z[A:Bb+1], self.xk[p], s, self.dz, gh[A:Bb+1])
            gE, LgE, SgE = np.empty(L), np.empty(L), np.empty(L)
            ew._tails(gh, A, Bb, X, gE, LgE, SgE)
            uE = (self.z[A-X:Bb+X+1]-self.xk[p])/s
            self.cache[p] = (gE, LgE, SgE, uE, dr.ndtr(uE), dr.ndtr(-uE), dr.phi(uE))
        self.g1 = np.empty(self.G); dr._kernel(self.z, self.xk[0], s, self.dz, self.g1)
        self.g1w = self.g1[self.a[1]:self.b[1]+1].copy()
        self.cache_s = time.perf_counter()-t0
        log(f"DESIGN {name} G = {self.G}, support window nodes {int(self.w[1:].min())}-{self.WMAX}, cache built in {self.cache_s:.1f} s")
        self.J_sod = None

    def sod_reference(self):
        """Part A on the SoD table: dispatches to the accepted operator (evaluator_fast), bitwise."""
        if self.J_sod is None:
            rec, J, r = ew.evaluate_table(self.name, ew.ConstTable(ew.SOD_M, ew.SOD_M))
            self.rec_sod, self.J_sod, self.W_sod = rec, J, r['W']
        return self.rec_sod, self.J_sod, self.W_sod

    def gamma(self, obj):
        """(gamma_k for k = 1..K, gbar = mean_{k>=2} gamma_k, gamma_1)."""
        if obj == 'S': return np.ones(K), 1.0, 1.0
        _, J, W = self.sod_reference()
        Lc = cholesky(J, lower=True)
        Y = solve_triangular(Lc, self.Hk.T, lower=True)
        g = np.sum(Y*Y, axis=0)/s**2
        return g, float(g[1:].mean()), float(g[0])

    def dp(self, gam, lam, codes):
        """backward induction at price lam; writes the argmax codes into codes[(K-1), WMAX]; returns (sum_i ghat_1 V_2, nbad)."""
        Vn = np.zeros(1); an, bn = 0, -1; nbad = 0
        L = self.WMAX+2*X; VnE, gvE, LgvE, SgvE = (np.empty(L) for _ in range(4))
        act = np.empty(self.WMAX, np.uint16)
        for p in range(K-1, 0, -1):
            A, Bb = int(self.a[p]), int(self.b[p]); nw = Bb-A+1
            gE, LgE, SgE, uE, PP, PN, PH = self.cache[p]
            _value_ext(Vn, an, bn, A, nw, gE, VnE, gvE, LgvE, SgvE)
            Vout = np.empty(nw)
            nbad += _dp_step(nw, gE, LgE, SgE, gvE, LgvE, SgvE, VnE, uE, PP, PN, PH, float(gam[p]), float(lam), Vout, act)
            codes[p-1, :nw] = act[:nw]; codes[p-1, nw:] = PAD
            Vn, an, bn = Vout, A, Bb
        assert an == self.a[1] and bn == self.b[1]
        return float(self.g1w@Vn), nbad

    def table(self, codes):
        return ew.CodeTable(codes, self.a[1:], self.w[1:])


# --- lambda scan with checkpoints ------------------------------------------------------------------------------------
def _atomic_json(obj, path):
    tmp = path+'.tmp'; json.dump(obj, open(tmp, 'w'), indent=1); os.replace(tmp, path)

def case_paths(case):
    tag = case.replace('/', '_')
    return tag, os.path.join(ew.RES, f'scan_{tag}.json'), os.path.join(ew.LOGS, f'scan_{tag}.log')

def mono_check(runs):
    """N = E[N] - 1 non-increasing in lambda over all evaluated runs: returns the worst relative violation and the pair."""
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

class Scan:
    def __init__(self, des, obj, log):
        self.des, self.obj, self.case = des, obj, f"{des.name}/{obj}"
        self.tag, self.jpath, _ = case_paths(self.case); self.log = log
        self.gam, self.gbar, self.g1 = des.gamma(obj)
        self.state = json.load(open(self.jpath)) if os.path.exists(self.jpath) else dict(case=self.case, runs=[], status='running')
        self.memo = {r['lam']: r for r in self.state['runs']}
        self.state.update(gbar=self.gbar, gamma1=self.g1, B=B)
        _, J_sod, W_sod = des.sod_reference()
        self.ld_sod = float(np.linalg.slogdet(J_sod)[1]); self.F_sod = float(self.gam[1:]@W_sod[1:])
        self.N_sod = float(des.rec_sod['EN'])-1.0
        log(f"CASE {self.case} gbar {self.gbar!r} gamma_1 {self.g1!r} F(SoD table) {self.F_sod!r} logdet(SoD) {self.ld_sod!r}")
        if obj == 'W':
            log(f"CASE {self.case} sum_(k>=1) gamma_k w_k(SoD) = tr(J_SoD^-1 J_SoD) {float(self.gam@W_sod)!r} (should be 9)")

    def run(self, lam, phase):
        """one DP + forward at lam (memoised by the exact float value); returns the run record."""
        if lam in self.memo:
            r = self.memo[lam]
            if os.path.exists(r['table']):
                self.log(f"LAM {self.case} run={r['run']:02d} phase={phase} lam={lam!r} (resumed from checkpoint) feasible={r['feasible']}")
                return r
        n = len(self.state['runs'])
        if n >= MAX_RUNS: raise RunCap()
        des = self.des; j = n+1
        tpath = os.path.join(TABLES, f"{self.tag}_r{j:02d}.u16")
        codes = np.lib.format.open_memmap(tpath+'.tmp.npy', mode='w+', dtype=np.uint16, shape=(K-1, des.WMAX))
        t0 = time.perf_counter(); dsum, nbad = des.dp(self.gam, lam, codes); t_dp = time.perf_counter()-t0
        codes.flush(); del codes; os.replace(tpath+'.tmp.npy', tpath+'.npy')
        codes = np.load(tpath+'.npy', mmap_mode='r')
        t0 = time.perf_counter()
        rec, J, rr = ew.evaluate_table(des.name, des.table(codes))
        t_fw = time.perf_counter()-t0
        W, Q = rr['W'], rr.get('Q')
        F = float(self.gam[1:]@W[1:]); EN = float(rec['EN']); N = EN-1.0
        dual = self.g1+dsum+lam*(B-1.0)
        r = dict(run=j, phase=phase, lam=lam, lam_over_gbar=lam/self.gbar, dual=dual, EN=EN, N=N, S=float(rec['S']), F=F,
                 feasible=bool(EN <= B), slack=B-EN, logdet=float(rec['logdet']), dmD=float(rec['logdet'])-self.ld_sod,
                 mass_dev_max=float(rec['mass_dev_max']), min_m=float(rec['min_m']),
                 assert_fail=int((~rr['assert_ok']).sum()), nonfinite_V=int(nbad), path=rec['path'],
                 dp_s=t_dp, fwd_s=t_fw, table=tpath+'.npy',
                 dp_minus_policy=dsum-(F-lam*N), policy_minus_sod=(F-lam*N)-(self.F_sod-lam*self.N_sod))
        np.savez(os.path.join(RUNS, f"{self.tag}_r{j:02d}.npz"), W=W, Q=(Q if Q is not None else np.full(K, np.nan)), J=J)
        self.state['runs'].append(r); self.memo[lam] = r
        mv = mono_check(self.state['runs']); self.state['mono_worst'] = mv[0]; self.state['mono_pair'] = mv[1]
        _atomic_json(self.state, self.jpath)
        self.log(f"LAM {self.case} run={j:02d} phase={phase} lam={lam!r} lam/gbar={lam/self.gbar!r} dual={dual!r} EN={EN!r} "
                 f"S={r['S']!r} F={F!r} feasible={r['feasible']} slack={r['slack']!r} dmD={r['dmD']!r} dp_s={t_dp:.1f} fwd_s={t_fw:.1f} "
                 f"[diag: V-policy {r['dp_minus_policy']!r}, policy-SoD {r['policy_minus_sod']!r}]")
        if mv[0] > MONO_TOL:
            raise MonoStop(mv)
        return r

    def scan(self):
        g = self.gbar; note = []
        try:
            lam = 5.0*g; r = self.run(lam, 'start')
            if not r['feasible']:
                while not r['feasible']:
                    lam = 2.0*lam
                    if lam > 100.0*g:
                        self.state['status'] = 'no feasible policy in the lambda range'
                        self.state['message'] = '在规定的 λ 扫描范围内未找到预算可行策略，停止本次设计'
                        self.log(f"STOP {self.case} {self.state['message']}"); _atomic_json(self.state, self.jpath); return self.state
                    r = self.run(lam, 'double')
            lam_f = lam; lam_i = None
            while True:
                lam = 0.8*lam
                r = self.run(lam, 'scan')
                if not r['feasible']: lam_i = lam; break
                lam_f = lam
                if lam < 1e-3*g:
                    note.append('在规定的 λ 扫描范围内未找到超预算端点'); break
            if lam_i is not None:
                for _ in range(N_BISECT):
                    mid = 0.5*(lam_f+lam_i)
                    r = self.run(mid, 'bisect')
                    if r['feasible']: lam_f = mid
                    else: lam_i = mid
            self.state['status'] = 'done'
        except RunCap:
            note.append('到限'); self.state['status'] = 'run cap (40 DP runs) reached'
        except MonoStop as e:
            self.state['status'] = 'monotonicity violation'; self.state['message'] = f'N(lambda) increased by relative {e.args[0][0]!r} between runs {e.args[0][1]}'
            self.log(f"STOP {self.case} monotonicity violation {e.args[0]!r}; no final policy (never worked around by another lambda)")
            _atomic_json(self.state, self.jpath); return self.state
        fin = select_final(self.state['runs'])
        self.state['notes'] = note; self.state['final_run'] = fin['run'] if fin else None
        self.state['min_dual'] = min(r['dual'] for r in self.state['runs'])
        self.state['lam_bracket'] = [lam_f, lam_i]
        _atomic_json(self.state, self.jpath)
        if fin:
            self.log(f"FINAL {self.case} run={fin['run']:02d} lam={fin['lam']!r} F={fin['F']!r} EN={fin['EN']!r} slack={fin['slack']!r} "
                     f"S={fin['S']!r} dmD={fin['dmD']!r}; DP runs {len(self.state['runs'])}; notes {note}")
        self.log(f"DUAL {self.case} 已评估 λ 中的最小对偶值 {self.state['min_dual']!r}")
        return self.state

class RunCap(Exception): pass
class MonoStop(Exception): pass


# --- B4 checks on the final tables, packed copies --------------------------------------------------------------------
def final_checks(log):
    out = {}
    designs = {}
    for case in CASES:
        tag, jpath, _ = case_paths(case)
        if not os.path.exists(jpath): log(f"B4 {case} no scan record"); continue
        st = json.load(open(jpath)); fr = st.get('final_run')
        if fr is None: log(f"B4 {case} no final policy (status {st.get('status')})"); continue
        r = [x for x in st['runs'] if x['run'] == fr][0]
        name, obj = case.split('/')
        if name not in designs: designs = {name: Design(name, log)}
        des = designs[name]; gam, gbar, g1 = des.gamma(obj)
        codes = np.load(r['table'], mmap_mode='r'); tab = des.table(codes)
        t0 = time.perf_counter()
        fw = ew.forward(des.z, des.dz, s, des.xk, des.a, des.b, tab, gam=gam)
        bw = ew.backward(des.z, des.dz, s, des.xk, des.a, des.b, tab, gam)
        t_c = time.perf_counter()-t0
        Ff = fw['Fgam']; Nf = fw['EN']-1.0
        eF = abs(bw['Fc']-Ff)/abs(Ff); eN = abs(bw['Nq']-Nf)/abs(Nf)
        res = dict(run=fr, lam=r['lam'], F_forward=Ff, F_backward=bw['Fc'], F_rel=eF, N_forward=Nf, N_backward=bw['Nq'], N_rel=eN,
                   fb_pass=bool(eF <= 1e-10 and eN <= 1e-10), EN=fw['EN'], budget_pass=bool(fw['EN'] <= B), slack=B-fw['EN'],
                   utilisation_target_met=bool(B-fw['EN'] <= 0.5), EN_scan=r['EN'], EN_same_as_scan=bool(fw['EN'] == r['EN']),
                   mono_worst=st.get('mono_worst'), runs=len(st['runs']), notes=st.get('notes'), status=st.get('status'),
                   check_s=t_c)
        packed = os.path.join(ew.RES, f"table_{tag}.npz")
        np.savez_compressed(packed, codes=np.asarray(codes), start=des.a[1:], width=des.w[1:], lam=r['lam'], run=fr,
                            note=np.array('codes[r, o]: action of step k = r + 2 at node start[r] + o (o < width[r]); '
                                          'code = i_- * 20 + i_+, (M^-, M^+) = (48 (i_- + 1), 48 (i_+ + 1)) nodes of sigma/24; pad 65535'))
        chk = np.load(packed)
        res['packed'] = os.path.relpath(packed, HERE); res['packed_bytes'] = os.path.getsize(packed)
        res['packed_roundtrip_equal'] = bool(np.array_equal(chk['codes'], np.asarray(codes)))
        out[case] = res
        for k, v in res.items(): log(f"B4 {case} {k} {v!r}")
    _atomic_json(out, os.path.join(ew.RES, 'final_checks.json'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['scan', 'final', 'time']); ap.add_argument('--case', action='append')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    numba.set_num_threads(args.threads)
    if args.stage == 'final':
        sys.stdout = dr.Tee(os.path.join(ew.LOGS, 'final_checks.log'))
        print(f"=== trigger_dp.py final (B4: forward-backward consistency, budget; packed tables); threads {args.threads}")
        _warmup(); final_checks(print); return
    cases = args.case or list(CASES)
    for case in cases:
        tag, jpath, lpath = case_paths(case)
        sys.stdout = dr.Tee(lpath if args.stage == 'scan' else os.path.join(ew.LOGS, f'time_{tag}.log'), 'a')
        print(f"=== trigger_dp.py {args.stage} {case} started {time.strftime('%Y-%m-%d %H:%M:%S')}; numba threads {numba.get_num_threads()}; "
              f"B = {B!r}; actions 400 (M^+- = 48..960); support R = 88 sigma")
        t0 = time.perf_counter(); _warmup(); print(f"WARMUP {time.perf_counter()-t0:.1f} s")
        name, obj = case.split('/')
        des = Design(name, print)
        if args.stage == 'time':
            gam, gbar, g1 = des.gamma(obj)
            codes = np.empty((K-1, des.WMAX), np.uint16)
            t0 = time.perf_counter(); dsum, nbad = des.dp(gam, 5.0*gbar, codes); t1 = time.perf_counter()-t0
            print(f"TIME {case} one DP at lambda_0 = 5 gbar: {t1:.1f} s (nonfinite V {nbad}); dual part {dsum!r}")
            continue
        t0 = time.perf_counter(); st = Scan(des, obj, print).scan()
        print(f"END {case} status {st['status']}; wall {time.perf_counter()-t0:.1f} s; DP runs {len(st['runs'])}")
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
