"""Round 8b, Part A3: the correctness checks of round 8 v3.1 §3 items 1-4 re-run under the round-8b convergence acceptance
(spec reports_CN/plan_round8b_CN.md v2, frozen). Every number goes to logs/partA_checks_r8b.log (one line per item) and
results/partA_checks.json.

Seeds: only the engineering-test seeds listed in predictions_round8b.md (round 8's A3 seeds 92000000 + ..., the archived
§8.4 record 20261014, restart seed 30261028 for the estimator calls, i.e. r = 0 as in round 8's A3). No official seed of this
round is used here.

A3-1 estimator (estimator_r8b: round 8's estimator, acceptance = Newton decrement <= 1e-5 with result.success and Cholesky)
  (a) complete data: the returned value equals least squares, max_i |theta_i - theta_LS,i| / s_i <= 1e-8 (round 8 DECISION
      A-8), and the call is accepted (converged, not a fallback);
  (b) one constant-SoD record of the archived §8.4 ML run (stage2/final_theta_mle.npz 'UNIF_mle' row 0; identity confirmed by
      LS = archived 'UNIF_ls' row 0): the returned estimate within 1e-3 statistical standard errors of the archived one,
      sqrt(dtheta^T H_obs dtheta) <= 1e-3, and accepted;
  (c) analytic vs numerical gradient of the joint likelihood under an asymmetric, (k, r)-varying random window table at
      three random theta: ||g_an - g_num||_inf / ||g_an||_inf <= 1e-6 (Richardson-extrapolated central differences);
  (d) [supporting] the extended likelihood equals mc84.py's nll_grad_hess on the SoD record; Hessian vs numerical gradient.
A3-2 design chain: theta_0 as the "estimate" (test mode, no sentinel), T4 (round7/results/table_T4.npz) bit for bit.
A3-3 isolation: round 8's static import graph of design_r8 and fresh-interpreter check (isolation_r8, unchanged); the same
     static and fresh-interpreter checks for the round-8b estimator module; sentinel positive control; one guarded design.
A3-4 conservation / non-negativity / support assertions over every evaluator call made here.
Supporting regressions (as in round 8): declared knowledge, verbatim mirror blocks byte-equal (57/57), post-hoc evaluator vs
round 7's sigma/120 archive at theta_0, table simulator = mc83.sod for a constant table, lookup = round 7 / round 5.

Usage (from this directory): python3 checks_r8b.py [--threads 4]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse, symtable, ast, subprocess, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); PKG = os.path.dirname(EV)
R8 = os.path.join(EV, 'round8')
for _p in (R8, HERE):
    if _p in sys.path: sys.path.remove(_p)
    sys.path.insert(0, _p)                                            # round8b first, then round8
import numpy as np

LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)
PROBE = 92000000                        # round 8's A3 seeds (engineering-test seed list)
SEED_ARCH = 20261014                    # the archived §8.4 record
RESTART_A3 = 30261028                   # restart seed of the estimator calls in A3 (r = 0, as in round 8's A3)
ALLOWED_EST = ('estimator_r8b', 'estimator_r8', 'declared', 'tables_r8', 'r8k_dr', 'r8k_ef', 'r8k_ew', 'r8k_td')


class Tee:
    def __init__(self, path): self.f = open(path, 'w'); self.o = sys.stdout
    def write(self, m): self.o.write(m); self.f.write(m); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()

def jl(v):
    if isinstance(v, dict): return {k: jl(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jl(x) for x in v]
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    return v


def static_check_8b(iso, start, allowed):
    """round 8's static import-graph check (isolation_r8: same forbidden module / name / call lists) for a module that lives in
    round8b/; repo-local imports are resolved in round8b/ first, then round8/."""
    def locate(m):
        for d in (HERE, R8):
            p = os.path.join(d, m+'.py')
            if os.path.exists(p): return p
        return None
    seen, third, todo = set(), set(), [start]
    while todo:
        m = todo.pop()
        if m in seen: continue
        seen.add(m)
        _, mods = iso._imports(locate(m))
        for x in mods:
            if locate(x): todo.append(x)
            elif x in iso.FORBIDDEN_MODULES: seen.add(x)
            else: third.add(x)
    local = sorted(seen); bad_mod = [m for m in local if m not in allowed]
    bad_third = sorted(m for m in third if m not in iso.THIRD_OK | iso.STDLIB_OK)
    bad_names = {}
    for m in local:
        p = locate(m)
        if p is None: continue
        hits = set()
        for n in ast.walk(ast.parse(open(p).read())):
            if isinstance(n, ast.Name) and (n.id in iso.FORBIDDEN_NAMES or n.id in iso.FORBIDDEN_CALLS): hits.add(n.id)
            elif isinstance(n, ast.Attribute) and n.attr in iso.FORBIDDEN_NAMES: hits.add('.'+n.attr)
            elif isinstance(n, ast.alias) and n.name.split('.')[0] in iso.FORBIDDEN_NAMES | {'importlib'}: hits.add('import '+n.name)
        if hits: bad_names[m] = sorted(hits)
    return dict(start=start, local=local, third_party=sorted(third), outside_allowlist=bad_mod, bad_third_party=bad_third,
                forbidden_names=bad_names, passed=not bad_mod and not bad_third and not bad_names)

def dynamic_check_8b(iso, start, allowed):
    code = (f"import sys, json; sys.path.insert(0, {R8!r}); sys.path.insert(0, {HERE!r}); import {start}; "
            "print(json.dumps(sorted([(n, getattr(m, '__file__', None)) for n, m in list(sys.modules.items())])))")
    out = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, cwd='/', timeout=600)
    if out.returncode != 0: return dict(passed=False, error=out.stderr[-2000:])
    mods = json.loads(out.stdout.strip().split('\n')[-1])
    repo = os.path.dirname(os.path.dirname(PKG))
    loaded_forbidden = [n for n, f in mods if n.split('.')[0] in iso.FORBIDDEN_MODULES]
    repo_local = sorted({n for n, f in mods if f and os.path.abspath(f).startswith(repo+os.sep)})
    outside = [n for n in repo_local if n not in allowed]
    return dict(passed=not loaded_forbidden and not outside, repo_local_loaded=repo_local, forbidden_loaded=loaded_forbidden,
                outside_allowlist=outside, n_modules=len(mods))


def main(threads):
    import numba; numba.set_num_threads(threads)
    out = {}; P = print
    import declared as dk, estimator_r8b as eb, tables_r8 as tb, isolation_r8 as iso, build_kernels as bk
    import design_r8 as dz
    import truth_r8 as tr, posthoc_r8 as ph
    er, rb, ew, dr = ph.er, ph.rb, ph.ew, ph.dr
    import scipy
    P(f"=== checks_r8b.py (round 8b Part A3, acceptance = Newton decrement <= {eb.DECR_TOL:g} with result.success and Cholesky) "
      f"started {time.strftime('%Y-%m-%d %H:%M:%S')}; python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, "
      f"numba {numba.__version__}; numba threads {numba.get_num_threads()}")
    P(f"A0 modules: estimator_r8b {eb.__file__}; estimator_r8 {eb.e8.__file__}; design_r8 {dz.__file__}; truth_r8 {tr.__file__}")
    t0 = time.perf_counter(); dz.warmup(); ph.warmup(); P(f"A0 warm-up {time.perf_counter()-t0:.1f} s")
    s = dk.s; SC = dk.SCALE; TH0 = tr.THETA0
    cons = dict(mass_dev_max=0.0, min_m=np.inf, n_eval=0, assert_fail=0)
    def note_eval(rec):
        cons['mass_dev_max'] = max(cons['mass_dev_max'], float(rec['mass_dev_max'])); cons['min_m'] = min(cons['min_m'], float(rec['min_m']))
        cons['n_eval'] += 1; cons['assert_fail'] += int(rec.get('assert_fail', 0))

    # ---------------- supporting: declared knowledge, verbatim blocks, generated globals (round 8, unchanged) -----------------
    xU, HU = dr.input_arrays('U')
    dec = dict(K=dk.K == dr.K == tr.K, t=np.array_equal(dk.t, dr.t), w=np.array_equal(dk.w, dr.w), a0=dk.a0 == dr.a0,
               D=dk.D == dr.D == tr.D, s=dk.s == dr.s == tr.s, B=dk.B == ew.B_BUDGET == er.B,
               H=np.array_equal(dk.H(), HU) and np.array_equal(dk.H(), tr.H_MC83),
               X0_eq_dr=np.array_equal(tr.X0, xU), x_polar_theta0=np.array_equal(dk.x_polar(TH0), xU),
               grid=(dk.N_GRID, dk.SOD_M, dk.STEP_M, dk.N_ACT1, dk.X24) == (ew.N_GRID, ew.SOD_M, ew.STEP_M, ew.N_ACT1, ew.X24),
               R_SUP=dk.R_SUP == ew.R_SUP, H_theta0_close=float(np.max(np.abs(dk.H()@TH0-xU))/s))
    dec['pass'] = all(v for k, v in dec.items() if k != 'H_theta0_close')
    for k, v in dec.items(): P(f"R-DECL {k} {v!r}")
    man = json.load(open(os.path.join(R8, 'kernels_manifest.json'))); vb = dict(blocks=0, equal=0, bad=[])
    for mod, info in man['modules'].items():
        text = open(os.path.join(R8, info['file'])).read()
        for b in info['blocks']:
            lo, hi, txt = bk.extract(b['source'], b['name']); vb['blocks'] += 1
            ok = [lo, hi] == b['lines'] and bk.sha(txt) == b['sha256'] and txt in text
            vb['equal'] += ok
            if not ok: vb['bad'].append((mod, b['name']))
    import builtins; undefined = {}
    for mod in man['modules']:
        src = open(os.path.join(R8, mod+'.py')).read(); top = symtable.symtable(src, mod, 'exec')
        defined = {sy.get_name() for sy in top.get_symbols() if sy.is_assigned() or sy.is_imported()}
        def walk(tab):
            for ch in tab.get_children():
                for sy in ch.get_symbols():
                    if sy.is_global() and sy.is_referenced() and sy.get_name() not in defined and not hasattr(builtins, sy.get_name()):
                        undefined.setdefault(mod, set()).add(sy.get_name())
                walk(ch)
        walk(top)
    vb['undefined_globals'] = {k: sorted(v) for k, v in undefined.items()}
    vb['pass'] = vb['equal'] == vb['blocks'] and not vb['bad']
    P(f"R-VERB round-8 verbatim mirror blocks re-extracted from the frozen sources and byte-equal: {vb['equal']}/{vb['blocks']} -> "
      f"{'PASS' if vb['pass'] else 'FAIL'}; round 8b adds no copy (every round-8 module is imported unchanged)")
    P(f"R-VERB global names used by functions but not defined in the generated module (legacy branches never called): {vb['undefined_globals']}")
    out['declared'] = dec; out['verbatim'] = vb

    # ---------------- A3-2 design-chain regression (test mode: theta_0 as the estimate, no sentinel) -------------------------
    t0 = time.perf_counter(); d0 = dz.design(TH0, 'test_theta0', log=P); t_d0 = time.perf_counter()-t0
    T4 = np.load(os.path.join(EV, 'round7', 'results', 'table_T4.npz'))
    st6 = json.load(open(os.path.join(EV, 'round6', 'results', 'scan_U_W_m020.json')))
    import evaluator_fast as ef
    recF, JF, _ = ef.evaluate_fast('U', 24)
    a2 = dict(status=d0['status'], codes_equal=bool(np.array_equal(d0['codes'], T4['codes'])), start_equal=bool(np.array_equal(d0['start'], T4['start'])),
              width_equal=bool(np.array_equal(d0['width'], T4['width'])), lam_equal=d0['lam'] == float(T4['lam']), lam=d0['lam'],
              z0_equal=d0['z0'] == er.Z0, h_equal=d0['h'] == er.H24, G_equal=d0['G'] == er.G24,
              EN_hat=d0['EN_hat'], EN_hat_equal_T4_sigma24=d0['EN_hat'] == 262.7986258567231,
              J_sod_bitwise_evaluator_fast=bool(np.array_equal(d0['J_sod_hat'], JF)), gbar=d0['gbar'], gbar_equal_scan=d0['gbar'] == st6['gbar'],
              wall_s=t_d0, t_dp_s=d0['t_dp_s'])
    a2['pass'] = all(a2[k] for k in ('codes_equal', 'start_equal', 'width_equal', 'lam_equal', 'z0_equal', 'h_equal', 'G_equal',
                                     'EN_hat_equal_T4_sigma24', 'J_sod_bitwise_evaluator_fast', 'gbar_equal_scan')) and a2['status'] == 'ok'
    for k, v in a2.items(): P(f"A3-2 {k} {v!r}")
    P(f"A3-2 design-chain regression (T4 bit for bit): {'PASS' if a2['pass'] else 'FAIL'}")
    cons['mass_dev_max'] = max(cons['mass_dev_max'], d0['mass_dev_max_hat'], d0['mass_dev_max_sod_hat']); cons['min_m'] = min(cons['min_m'], d0['min_m_hat'], d0['min_m_sod_hat']); cons['n_eval'] += 2
    out['A3_2'] = a2

    # ---------------- supporting: post-hoc evaluator vs round 7 archive (sigma/120, theta_0) --------------------------------
    A7 = [json.loads(l) for l in open(os.path.join(EV, 'round7', 'results', 'archive_regrade.jsonl'))]
    ref = {r['cand']: r for r in A7 if r['pid'] == 'theta_0' and r['f'] == 5}
    tab0 = tb.SeqTable.from_design(d0, 'theta0')
    t0 = time.perf_counter(); recT, JT, rT = ph.eval_table(tab0); t_ph = time.perf_counter()-t0; note_eval(recT)
    recS, JS, rS = ph.eval_sod_ref(); note_eval(recS)
    phr = dict(T4_EN=recT['EN'], T4_EN_equal=recT['EN'] == ref['T4']['EN'], T4_S_equal=recT['S'] == ref['T4']['S'],
               T4_logdet_equal=recT['logdet'] == ref['T4']['logdet'], T4_J_equal=bool(np.array_equal(JT, np.array(ref['T4']['J']))),
               grid_n0=recT['grid_n0'], boundary_gap=recT['boundary_gap'], b1=recS['EN'], b1_equal=recS['EN'] == ref['SoD']['EN'],
               J1_equal=bool(np.array_equal(JS, np.array(ref['SoD']['J']))), t_posthoc_s=t_ph)
    phr['pass'] = all(phr[k] for k in ('T4_EN_equal', 'T4_S_equal', 'T4_logdet_equal', 'T4_J_equal', 'b1_equal', 'J1_equal'))
    for k, v in phr.items(): P(f"R-POST {k} {v!r}")
    out['posthoc_regression'] = phr

    # ---------------- supporting: simulator and lookup ---------------------------------------------------------------------
    const = dict(d0); const['codes'] = np.full_like(d0['codes'], ew.SOD_CODE); tabc = tb.SeqTable.from_design(const, 'const480')
    n_same = 0; n_ev = 0
    for i in range(100):
        e = np.random.default_rng(PROBE+i).standard_normal(dk.K)
        z1, e1, r1 = tr.sim_sod(e); z2, e2, r2 = tr.sim_table(e, tabc)
        n_same += bool(np.array_equal(e1, e2) and np.array_equal(r1, r2, equal_nan=True)); n_ev += int(e1.sum())
    rng = np.random.default_rng(PROBE+500); kk = rng.integers(2, dk.K+1, 20000)
    rr = rng.uniform(tab0.z0-2*dk.D, tab0.z0+(d0['G']+50)*tab0.h, 20000)
    mm_v, mp_v = tab0.lookup_vec(kk, rr); sc = [tab0.lookup(int(k), float(r)) for k, r in zip(kk, rr)]
    cell, _ = er.cell_index(rr, tab0.z0, tab0.h); lt = rb.LookupTable(tab0.codes, tab0.start, tab0.width)
    ref_m = np.array([[int(x) for x in lt.rows(int(k)-1, int(c), int(c))[j]] for k, c in zip(kk, cell) for j in (0, 1)]).reshape(-1, 2)
    sim = dict(const480_equal_mc83_sod=n_same, trajectories=100, events=n_ev,
               lookup_scalar_eq_vec=bool(np.array_equal(np.c_[mm_v, mp_v], np.array(sc))),
               lookup_eq_round7_cell_round5_table=bool(np.array_equal(np.c_[mm_v, mp_v], ref_m)), lookups=int(kk.size))
    sim['pass'] = n_same == 100 and sim['lookup_scalar_eq_vec'] and sim['lookup_eq_round7_cell_round5_table']
    for k, v in sim.items(): P(f"R-SIM {k} {v!r}")
    out['simulator'] = sim

    # ---------------- A3-1 estimator under the round-8b acceptance ---------------------------------------------------------
    a1 = {}
    def term(o, ref_theta, Hfun=None):
        rows = []
        for a in o['attempts']:
            row = dict(success=a['success'], message=a['message'], nit=a['nit'], grad_inf=a['grad_inf'], decrement=a['decrement'],
                       accepted=eb.accepted(a), dist_over_s=float(np.max(np.abs(np.array(a['theta'])-ref_theta)/SC)))
            if Hfun is not None:
                dth = np.array(a['theta'])-ref_theta; row['se_dist'] = float(np.sqrt(dth@Hfun(np.array(a['theta']))@dth))
            rows.append(row)
        return rows
    # (a) complete data
    e = np.random.default_rng(PROBE+1000).standard_normal(dk.K); z = tr.X0+s*e
    full = eb.StageData(np.ones(dk.K, bool), z, ('sod',))
    o = eb.estimate_stage1(full, RESTART_A3)
    th_ls = o['theta_LS']
    ra = dict(LS_usable=o['LS_usable'], converged=o['converged'], used=o['used'], fallback=o['fallback'],
              dist=float(np.max(np.abs(o['theta']-th_ls)/SC)), terminal=term(o, th_ls))
    ra['pass'] = bool(ra['dist'] <= 1e-8 and ra['converged'])
    a1['a'] = ra
    for k, v in ra.items(): P(f"A3-1a {k} {v!r}")
    # (b) archived §8.4 ML record
    arch = np.load(os.path.join(PKG, 'stage2', 'final_theta_mle.npz'))
    Z2 = np.random.default_rng(SEED_ARCH).standard_normal((2, dk.K))[0]
    z, eps, r = tr.sim_sod(Z2); rec = tr.record_from(z, eps)
    st1 = eb.StageData(rec['ev'], rec['zt'], ('sod',))
    o = eb.estimate_stage1(st1, RESTART_A3)
    Hf = lambda th: eb.nll_grad_hess_theta(st1, th)[2]
    dth = o['theta']-arch['UNIF_mle'][0]
    rb_ = dict(N=int(eps.sum()), LS_equal_archived=bool(np.array_equal(o['theta_LS'], arch['UNIF_ls'][0])),
               LS_maxdiff=float(np.max(np.abs(o['theta_LS']-arch['UNIF_ls'][0]))), converged=o['converged'], used=o['used'],
               fallback=o['fallback'], se_dist=float(np.sqrt(dth@Hf(o['theta'])@dth)), max_abs_over_s=float(np.max(np.abs(dth)/SC)),
               terminal=term(o, arch['UNIF_mle'][0], Hf))
    rb_['pass'] = bool(rb_['LS_equal_archived'] and rb_['se_dist'] <= 1e-3 and rb_['converged'])
    a1['b'] = rb_
    for k, v in rb_.items(): P(f"A3-1b {k} {v!r}")
    # (c) gradient under asymmetric (k, r)-varying windows
    rng = np.random.default_rng(PROBE+2000)
    rnd = dict(d0); rnd['codes'] = rng.integers(0, 400, size=d0['codes'].shape).astype(np.uint16)
    tabr = tb.SeqTable.from_design(rnd, 'random')
    e1 = np.random.default_rng(PROBE+2001).standard_normal(dk.K); e2 = np.random.default_rng(PROBE+2002).standard_normal(dk.K)
    zA, eA, _ = tr.sim_sod(e1); zB, eB, rB = tr.sim_table(e2, tabr)
    sA = eb.StageData(eA, np.where(eA, zA, np.nan), ('sod',)); sB = eb.StageData(eB, np.where(eB, zB, np.nan), ('table', tabr))
    joint = eb.stack([sA, sB])
    rc = dict(N1=sA.N, N2=sB.N, frac_asymmetric_windows=float(np.mean(sB.Mm != sB.Mp)),
              distinct_windows=int(len(set(zip(sB.Mm.tolist(), sB.Mp.tolist())))), points=[])
    for j in range(3):
        th = TH0+1e-4*SC*np.random.default_rng(PROBE+2100+j).standard_normal(dk.N_PAR)
        ob = eb.Objective(joint, th); f0, g = ob.f_g(np.zeros(dk.N_PAR))
        def cd(hh):
            gn = np.empty(dk.N_PAR)
            for i in range(dk.N_PAR):
                ei = np.zeros(dk.N_PAR); ei[i] = hh
                gn[i] = (ob.f_g(ei)[0]-ob.f_g(-ei)[0])/(2*hh)
            return gn
        g1, g2 = cd(1e-8), cd(2e-8); gn = (4*g1-g2)/3
        rc['points'].append(dict(theta_offset_over_s=((th-TH0)/SC).tolist(), f=f0, g_inf=float(np.max(np.abs(g))),
                                 rel_err=float(np.max(np.abs(g-gn))/np.max(np.abs(g))),
                                 rel_err_plain_cd=float(np.max(np.abs(g-g1))/np.max(np.abs(g)))))
    rc['pass'] = all(p['rel_err'] <= 1e-6 for p in rc['points'])
    a1['c'] = rc
    for k, v in rc.items(): P(f"A3-1c {k} {v!r}")
    # (d) extension regression vs mc84.py, Hessian vs gradient differences (supporting)
    src84 = open(os.path.join(PKG, 'mc84.py')).read()
    defs84 = src84.split('rng = np.random.default_rng(seed); Z = rng.standard_normal((R, K))')[1].split('t0 = time.time(); TH')[0]
    from scipy.special import log_ndtr, ndtr
    ns = dict(np=np, H=dk.H(), K=dk.K, D=dk.D, s=s, log_ndtr=log_ndtr, ndtr=ndtr); exec(defs84, ns)
    rd = dict(points=[])
    for j in range(3):
        th = arch['UNIF_mle'][0]+1e-4*SC*np.random.default_rng(PROBE+2200+j).standard_normal(dk.N_PAR)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f84, g84, H84 = ns['nll_grad_hess'](th, z, eps, r)
        f8, g8, H8 = eb.nll_grad_hess_theta(st1, th)
        ob = eb.Objective(st1, th); hh = 1e-8; Hn = np.empty((9, 9))
        for i in range(9):
            ei = np.zeros(9); ei[i] = hh; Hn[:, i] = (ob.f_g(ei)[1]-ob.f_g(-ei)[1])/(2*hh)
        Hs = ob.hess_vartheta(np.zeros(9))
        rd['points'].append(dict(f_rel=abs(f8-f84)/abs(f84), g_rel=float(np.max(np.abs(g8-g84))/np.max(np.abs(g84))),
                                 H_rel=float(np.max(np.abs(H8-H84))/np.max(np.abs(H84))),
                                 H_vs_numgrad=float(np.max(np.abs(Hs-Hn))/np.max(np.abs(Hs)))))
    rd['pass'] = all(p['f_rel'] <= 1e-10 and p['g_rel'] <= 1e-10 and p['H_rel'] <= 1e-10 and p['H_vs_numgrad'] <= 1e-6 for p in rd['points'])
    a1['d'] = rd
    for k, v in rd.items(): P(f"A3-1d {k} {v!r}")
    a1['pass'] = bool(ra['pass'] and rb_['pass'] and rc['pass'])
    P(f"A3-1 estimator check under the round-8b acceptance (a, b, c): {'PASS' if a1['pass'] else 'FAIL'} "
      f"(a {ra['pass']}, b {rb_['pass']}, c {rc['pass']}; supporting d {rd['pass']})")
    out['A3_1'] = a1

    # ---------------- A3-3 isolation ---------------------------------------------------------------------------------------
    sc_ = iso.static_check('design_r8'); dc = iso.dynamic_check('design_r8')
    sc_e = static_check_8b(iso, 'estimator_r8b', ALLOWED_EST); dc_e = dynamic_check_8b(iso, 'estimator_r8b', ALLOWED_EST)
    iso.ACCESS_LOG.clear(); pos = []
    with iso.guard() as names:
        for nm, f in (('THETA0 + 1', lambda: tr.THETA0+1), ('np.asarray(X0)', lambda: np.asarray(tr.X0)), ('THETA0[0]', lambda: tr.THETA0[0]),
                      ('dr.Gj.real', lambda: dr.Gj.real), ('dr.input_arrays("U")', lambda: dr.input_arrays('U')),
                      ('rb.THETA0 @ H', lambda: dk.H()@rb.THETA0), ('er.I24["xbar"]', lambda: er.I24['xbar'])):
            try: f(); pos.append((nm, False))
            except iso.SentinelAccess: pos.append((nm, True))
    logged = sorted({n for n, _ in iso.ACCESS_LOG})
    pos_ok = all(v for _, v in pos) and len(iso.ACCESS_LOG) >= len(pos)
    restored = bool(np.array_equal(tr.THETA0, TH0) and isinstance(tr.X0, np.ndarray))
    iso.ACCESS_LOG.clear()
    th_g = TH0+0.3*s/np.max(np.abs(dk.H()@np.r_[1.0, np.zeros(8)]))*np.r_[1.0, np.zeros(8)]
    with iso.guard() as names:
        t0 = time.perf_counter(); dg = dz.design(th_g, 'guarded', log=P); t_g = time.perf_counter()-t0
    guarded_ok = len(iso.ACCESS_LOG) == 0 and dg['status'] == 'ok'
    cons['mass_dev_max'] = max(cons['mass_dev_max'], dg['mass_dev_max_hat'], dg['mass_dev_max_sod_hat']); cons['min_m'] = min(cons['min_m'], dg['min_m_hat'], dg['min_m_sod_hat']); cons['n_eval'] += 2
    a3 = dict(static=sc_, dynamic=dc, static_estimator_r8b=sc_e, dynamic_estimator_r8b=dc_e, guarded_holders=names, positive_control=pos,
              positive_control_pass=pos_ok, positive_control_logged=logged, restored_after_guard=restored, guarded_design_status=dg['status'],
              guarded_design_access_log=list(iso.ACCESS_LOG), guarded_design_EN_hat=dg.get('EN_hat'), guarded_design_wall_s=t_g)
    a3['pass'] = bool(sc_['passed'] and dc['passed'] and sc_e['passed'] and dc_e['passed'] and pos_ok and restored and guarded_ok)
    for k in ('local', 'third_party', 'outside_allowlist', 'bad_third_party', 'forbidden_names', 'passed'): P(f"A3-3 static design_r8 {k} {sc_[k]!r}")
    for k, v in dc.items(): P(f"A3-3 dynamic design_r8 {k} {v!r}")
    for k in ('local', 'third_party', 'outside_allowlist', 'bad_third_party', 'forbidden_names', 'passed'): P(f"A3-3 static estimator_r8b {k} {sc_e[k]!r}")
    for k, v in dc_e.items(): P(f"A3-3 dynamic estimator_r8b {k} {v!r}")
    P(f"A3-3 sentinel holders {names}")
    P(f"A3-3 sentinel positive control {pos}; holders logged {logged} -> {'PASS' if pos_ok else 'FAIL'}; originals restored {restored}")
    P(f"A3-3 guarded design at theta_0 + 0.3 sigma e1: status {dg['status']}, sentinel accesses {len(iso.ACCESS_LOG)}, E_hat[N] {dg.get('EN_hat')!r}")
    P(f"A3-3 isolation: {'PASS' if a3['pass'] else 'FAIL'}")
    out['A3_3'] = a3

    # ---------------- A3-4 assertions --------------------------------------------------------------------------------------
    a4 = dict(cons); a4['pass'] = bool(cons['mass_dev_max'] <= 1e-10 and cons['min_m'] >= 0 and cons['assert_fail'] == 0)
    for k, v in a4.items(): P(f"A3-4 {k} {v!r}")
    out['A3_4'] = a4
    out['all_required_pass'] = bool(a1['pass'] and a2['pass'] and a3['pass'] and a4['pass'])
    out['supporting_pass'] = bool(dec['pass'] and vb['pass'] and phr['pass'] and sim['pass'] and rd['pass'])
    out['seeds_used'] = dict(probe_base=PROBE, archived_record=SEED_ARCH, restart=RESTART_A3)
    P(f"A3 SUMMARY (round-8b acceptance): check 1 {'PASS' if a1['pass'] else 'FAIL'}, check 2 {'PASS' if a2['pass'] else 'FAIL'}, check 3 "
      f"{'PASS' if a3['pass'] else 'FAIL'}, check 4 {'PASS' if a4['pass'] else 'FAIL'}; supporting regressions "
      f"{'PASS' if out['supporting_pass'] else 'FAIL'}; all required {'PASS' if out['all_required_pass'] else 'FAIL'}")
    json.dump(jl(out), open(os.path.join(RES, 'partA_checks.json'), 'w'), indent=1)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--threads', type=int, default=4); a = ap.parse_args()
    sys.stdout = Tee(os.path.join(LOGS, 'partA_checks_r8b.log'))
    t0 = time.perf_counter(); o = main(a.threads); print(f"=== checks_r8b.py done in {time.perf_counter()-t0:.1f} s")
