"""Round 8, Part A3: correctness checks (spec reports_CN/plan_round8_CN.md v3.1, frozen; §3 items 1-4) and supporting
regressions. Every number goes to logs/partA_checks.log (one line per item) and results/partA_checks.json.

A3-1 estimator
  (a) complete data (every sample transmitted): the estimator's returned value equals least squares,
      max_i |theta_i - theta_LS,i| / s_i <= 1e-8 (DECISION A3-1); reported for both acceptance rules with the convergence flag,
      and for the optimiser's terminal iterates.
  (b) one constant-SoD record of the archived §8.4 ML run (stage2/final_theta_mle.npz 'UNIF_mle' row 0: uniform input, seed
      20261014 row 0, mc84.py's damped Newton): record identity confirmed by LS on the sent samples equal to the archived
      'UNIF_ls' row 0; pass iff the returned estimate is within 1e-3 statistical standard errors of the archived one,
      sqrt(dtheta^T H_obs dtheta) <= 1e-3 (DECISION A3-1). Reported for both rules.
  (c) analytic vs numerical gradient of the joint likelihood (a stage-1 SoD record + a stage-2 record under an asymmetric,
      (k, r)-varying random window table) at three random theta: ||g_an - g_num||_inf / ||g_an||_inf <= 1e-6 (Richardson
      extrapolated central differences in vartheta, steps 1e-8 and 2e-8).
  (d) [supporting] on a constant-SoD record the extended likelihood equals mc84.py's nll_grad_hess (exec'd verbatim):
      objective, gradient, Hessian relative <= 1e-10; Hessian vs numerical gradient differences <= 1e-6.
  (e) [diagnostic for DECISION E-1] floating-point floor of ||grad_vartheta l||_inf for the summed objective.
A3-2 design chain: theta_0 passed as the "estimate" (test mode, no sentinel): codes, start, width, lambda bit for bit equal
     to T4 (round7/results/table_T4.npz); z0 equal to round 7's frozen grid origin; designer's E[N] equal to T4's sigma/24
     value; J_SoD_hat bit for bit evaluator_fast; gbar equal to the round-6 scan record.
A3-3 isolation: static import graph of design_r8, fresh-interpreter import check, sentinel positive control (deliberate
     accesses must raise and be logged), one guarded design at a theta != theta_0 with an empty access log.
A3-4 conservation / non-negativity / support assertions over every evaluator call made here (they raise if violated).
Supporting regressions: declared knowledge bit for bit against the frozen sources; verbatim blocks re-extracted and equal;
global names of the generated modules defined; table simulator = mc83.sod for a constant (480, 480) table; scalar lookup =
vectorised lookup = round 7 cell_index + round 5 LookupTable; post-hoc evaluator = round 7's sigma/120 archive at theta_0 for
T4 and the SoD (b_1) bit for bit.

Usage (from this directory): python3 checks_r8.py [--threads 4]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse, symtable
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); PKG = os.path.dirname(EV)
sys.path.insert(0, HERE)
import numpy as np

LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)
PROBE = 92000000                        # probe seeds for synthetic records (disjoint from every seed of the pipeline)


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


def main(threads):
    import numba; numba.set_num_threads(threads)
    out = {}; P = print
    import declared as dk, estimator_r8 as est, tables_r8 as tb, isolation_r8 as iso, build_kernels as bk
    import design_r8 as dz
    import truth_r8 as tr, posthoc_r8 as ph
    er, rb, ew, dr = ph.er, ph.rb, ph.ew, ph.dr
    import scipy
    P(f"=== checks_r8.py (round 8 Part A3) started {time.strftime('%Y-%m-%d %H:%M:%S')}; python {sys.version.split()[0]}, numpy "
      f"{np.__version__}, scipy {scipy.__version__}, numba {numba.__version__}; numba threads {numba.get_num_threads()}")
    t0 = time.perf_counter(); dz.warmup(); ph.warmup(); P(f"A0 warm-up {time.perf_counter()-t0:.1f} s")
    s = dk.s; SC = dk.SCALE; TH0 = tr.THETA0
    cons = dict(mass_dev_max=0.0, min_m=np.inf, n_eval=0, assert_fail=0)
    def note_eval(rec):
        cons['mass_dev_max'] = max(cons['mass_dev_max'], float(rec['mass_dev_max'])); cons['min_m'] = min(cons['min_m'], float(rec['min_m']))
        cons['n_eval'] += 1; cons['assert_fail'] += int(rec.get('assert_fail', 0))

    # ---------------- supporting: declared knowledge, verbatim blocks, generated globals -------------------------------------
    xU, HU = dr.input_arrays('U')
    dec = dict(K=dk.K == dr.K == tr.K, t=np.array_equal(dk.t, dr.t), w=np.array_equal(dk.w, dr.w), a0=dk.a0 == dr.a0,
               D=dk.D == dr.D == tr.D, s=dk.s == dr.s == tr.s, B=dk.B == ew.B_BUDGET == er.B,
               H=np.array_equal(dk.H(), HU) and np.array_equal(dk.H(), tr.H_MC83),
               X0_eq_dr=np.array_equal(tr.X0, xU), x_polar_theta0=np.array_equal(dk.x_polar(TH0), xU),
               grid=(dk.N_GRID, dk.SOD_M, dk.STEP_M, dk.N_ACT1, dk.X24) == (ew.N_GRID, ew.SOD_M, ew.STEP_M, ew.N_ACT1, ew.X24),
               R_SUP=dk.R_SUP == ew.R_SUP, H_theta0_close=float(np.max(np.abs(dk.H()@TH0-xU))/s))
    dec['pass'] = all(v for k, v in dec.items() if k != 'H_theta0_close')
    for k, v in dec.items(): P(f"R-DECL {k} {v!r}")
    man = json.load(open(os.path.join(HERE, 'kernels_manifest.json'))); vb = dict(blocks=0, equal=0, bad=[])
    for mod, info in man['modules'].items():
        text = open(os.path.join(HERE, info['file'])).read()
        for b in info['blocks']:
            lo, hi, txt = bk.extract(b['source'], b['name']); vb['blocks'] += 1
            ok = [lo, hi] == b['lines'] and bk.sha(txt) == b['sha256'] and txt in text
            vb['equal'] += ok
            if not ok: vb['bad'].append((mod, b['name']))
    # global names used inside every function of the generated modules must be defined at module level or be builtins
    import builtins; undefined = {}
    for mod in man['modules']:
        src = open(os.path.join(HERE, mod+'.py')).read(); top = symtable.symtable(src, mod, 'exec')
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
    P(f"R-VERB verbatim blocks re-extracted from the frozen sources and byte-equal: {vb['equal']}/{vb['blocks']} -> {'PASS' if vb['pass'] else 'FAIL'}")
    P(f"R-VERB global names used by functions but not defined in the generated module (only legacy branches never called "
      f"may appear): {vb['undefined_globals']}")
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
    rr = rng.uniform(tab0.z0-2*dk.D, tab0.z0+(d0['G']+50)*tab0.h, 20000)            # includes cells outside the stored ranges
    mm_v, mp_v = tab0.lookup_vec(kk, rr); sc = [tab0.lookup(int(k), float(r)) for k, r in zip(kk, rr)]
    cell, _ = er.cell_index(rr, tab0.z0, tab0.h); lt = rb.LookupTable(tab0.codes, tab0.start, tab0.width)
    ref_m = np.array([[int(x) for x in lt.rows(int(k)-1, int(c), int(c))[j]] for k, c in zip(kk, cell) for j in (0, 1)]).reshape(-1, 2)
    sim = dict(const480_equal_mc83_sod=n_same, trajectories=100, events=n_ev,
               lookup_scalar_eq_vec=bool(np.array_equal(np.c_[mm_v, mp_v], np.array(sc))),
               lookup_eq_round7_cell_round5_table=bool(np.array_equal(np.c_[mm_v, mp_v], ref_m)),
               lookups=int(kk.size), outside_stored_range=int(np.sum((cell < tab0.start[kk-2]) | (cell > tab0.start[kk-2]+tab0.width[kk-2]-1))))
    sim['pass'] = n_same == 100 and sim['lookup_scalar_eq_vec'] and sim['lookup_eq_round7_cell_round5_table']
    for k, v in sim.items(): P(f"R-SIM {k} {v!r}")
    out['simulator'] = sim

    # ---------------- A3-1 estimator ----------------------------------------------------------------------------------------
    a1 = {}
    # (a) complete data
    e = np.random.default_rng(PROBE+1000).standard_normal(dk.K); z = tr.X0+s*e
    full = est.StageData(np.ones(dk.K, bool), z, ('sod',))
    o = est.estimate_stage1(full, 0)
    th_ls = o['theta_LS']
    ra = dict(LS_usable=o['LS_usable'])
    for rule in est.RULES:
        q = o['rules'][rule]
        ra[rule] = dict(converged=q['converged'], used=q['used'], fallback=q['fallback'], dist=float(np.max(np.abs(q['theta']-th_ls)/SC)))
    ra['terminal'] = [dict(success=a['success'], message=a['message'], grad_inf=a['grad_inf'], decrement=a['decrement'],
                           dist=float(np.max(np.abs(np.array(a['theta'])-th_ls)/SC))) for a in o['attempts']]
    ra['pass'] = all(ra[rule]['dist'] <= 1e-8 for rule in est.RULES)
    a1['a'] = ra
    for k, v in ra.items(): P(f"A3-1a {k} {v!r}")
    # (b) archived §8.4 ML record
    arch = np.load(os.path.join(PKG, 'stage2', 'final_theta_mle.npz'))
    Z2 = np.random.default_rng(20261014).standard_normal((2, dk.K))[0]
    z, eps, r = tr.sim_sod(Z2); rec = tr.record_from(z, eps)
    st1 = est.StageData(rec['ev'], rec['zt'], ('sod',))
    o = est.estimate_stage1(st1, 0)
    rb_ = dict(N=int(eps.sum()), LS_equal_archived=bool(np.array_equal(o['theta_LS'], arch['UNIF_ls'][0])),
               LS_maxdiff=float(np.max(np.abs(o['theta_LS']-arch['UNIF_ls'][0]))))
    for rule in est.RULES:
        q = o['rules'][rule]; dth = q['theta']-arch['UNIF_mle'][0]
        Hobs = est.nll_grad_hess_theta(st1, q['theta'])[2]
        rb_[rule] = dict(converged=q['converged'], used=q['used'], fallback=q['fallback'], se_dist=float(np.sqrt(dth@Hobs@dth)),
                         max_abs_over_s=float(np.max(np.abs(dth)/SC)))
    for j, a in enumerate(o['attempts']):
        dth = np.array(a['theta'])-arch['UNIF_mle'][0]; Hobs = est.nll_grad_hess_theta(st1, np.array(a['theta']))[2]
        rb_[f'terminal_{j+1}'] = dict(success=a['success'], grad_inf=a['grad_inf'], decrement=a['decrement'], se_dist=float(np.sqrt(dth@Hobs@dth)))
    rb_['pass_frozen'] = rb_['LS_equal_archived'] and rb_['frozen']['se_dist'] <= 1e-3
    rb_['pass_supp'] = rb_['LS_equal_archived'] and rb_['supp']['se_dist'] <= 1e-3
    rb_['pass'] = rb_['pass_frozen']
    a1['b'] = rb_
    for k, v in rb_.items(): P(f"A3-1b {k} {v!r}")
    # (c) gradient under asymmetric (k, r)-varying windows
    rng = np.random.default_rng(PROBE+2000)
    rnd = dict(d0); rnd['codes'] = rng.integers(0, 400, size=d0['codes'].shape).astype(np.uint16)
    tabr = tb.SeqTable.from_design(rnd, 'random')
    e1 = np.random.default_rng(PROBE+2001).standard_normal(dk.K); e2 = np.random.default_rng(PROBE+2002).standard_normal(dk.K)
    zA, eA, _ = tr.sim_sod(e1); zB, eB, rB = tr.sim_table(e2, tabr)
    sA = est.StageData(eA, np.where(eA, zA, np.nan), ('sod',)); sB = est.StageData(eB, np.where(eB, zB, np.nan), ('table', tabr))
    joint = est.stack([sA, sB])
    asym = float(np.mean(sB.Mm != sB.Mp))
    rc = dict(N1=sA.N, N2=sB.N, frac_asymmetric_windows=asym, distinct_windows=int(len(set(zip(sB.Mm.tolist(), sB.Mp.tolist())))), points=[])
    for j in range(3):
        th = TH0+1e-4*SC*np.random.default_rng(PROBE+2100+j).standard_normal(dk.N_PAR)
        ob = est.Objective(joint, th); f0, g = ob.f_g(np.zeros(dk.N_PAR))
        def cd(hh):
            gn = np.empty(dk.N_PAR)
            for i in range(dk.N_PAR):
                ei = np.zeros(dk.N_PAR); ei[i] = hh
                gn[i] = (ob.f_g(ei)[0]-ob.f_g(-ei)[0])/(2*hh)
            return gn
        g1, g2 = cd(1e-8), cd(2e-8); gn = (4*g1-g2)/3
        err = float(np.max(np.abs(g-gn))/np.max(np.abs(g)))
        rc['points'].append(dict(theta_offset_over_s=((th-TH0)/SC).tolist(), f=f0, g_inf=float(np.max(np.abs(g))), rel_err=err,
                                 rel_err_plain_cd=float(np.max(np.abs(g-g1))/np.max(np.abs(g))),
                                 max_component_rel=float(np.max(np.abs(g-gn)/np.maximum(np.abs(g), 1e-300)))))
    rc['pass'] = all(p['rel_err'] <= 1e-6 for p in rc['points'])
    a1['c'] = rc
    for k, v in rc.items(): P(f"A3-1c {k} {v!r}")
    # (d) extension regression vs mc84.py, Hessian vs gradient differences
    src84 = open(os.path.join(PKG, 'mc84.py')).read()
    defs84 = src84.split('rng = np.random.default_rng(seed); Z = rng.standard_normal((R, K))')[1].split('t0 = time.time(); TH')[0]
    from scipy.special import log_ndtr, ndtr
    ns = dict(np=np, H=dk.H(), K=dk.K, D=dk.D, s=s, log_ndtr=log_ndtr, ndtr=ndtr); exec(defs84, ns)
    import warnings
    rd = dict(points=[])
    for j in range(3):
        th = arch['UNIF_mle'][0]+1e-4*SC*np.random.default_rng(PROBE+2200+j).standard_normal(dk.N_PAR)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f84, g84, H84 = ns['nll_grad_hess'](th, z, eps, r)
        f8, g8, H8 = est.nll_grad_hess_theta(st1, th)
        ob = est.Objective(st1, th); hh = 1e-8; Hn = np.empty((9, 9))
        for i in range(9):
            ei = np.zeros(9); ei[i] = hh; Hn[:, i] = (ob.f_g(ei)[1]-ob.f_g(-ei)[1])/(2*hh)
        Hs = ob.hess_vartheta(np.zeros(9))
        rd['points'].append(dict(f_rel=abs(f8-f84)/abs(f84), g_rel=float(np.max(np.abs(g8-g84))/np.max(np.abs(g84))),
                                 H_rel=float(np.max(np.abs(H8-H84))/np.max(np.abs(H84))),
                                 H_vs_numgrad=float(np.max(np.abs(Hs-Hn))/np.max(np.abs(Hs)))))
    rd['pass'] = all(p['f_rel'] <= 1e-10 and p['g_rel'] <= 1e-10 and p['H_rel'] <= 1e-10 and p['H_vs_numgrad'] <= 1e-6 for p in rd['points'])
    a1['d'] = rd
    for k, v in rd.items(): P(f"A3-1d {k} {v!r}")
    # (e) floating-point floor of the frozen gradient threshold
    th_hat = np.array(o['rules']['supp']['theta'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g_raw = ns['nll_grad_hess'](th_hat, z, eps, r)[1]*SC
        ulp_g = []
        for i in range(9):
            tt = th_hat.copy(); tt[i] = np.nextafter(tt[i], np.inf)
            ulp_g.append(float(np.max(np.abs(ns['nll_grad_hess'](tt, z, eps, r)[1]*SC-g_raw))))
    Hs = est.Objective(st1, th_hat).hess_vartheta(np.zeros(9)); ev_ = np.linalg.eigvalsh(Hs)
    re = dict(grad_inf_raw_at_estimate=float(np.max(np.abs(g_raw))), grad_change_one_ulp_per_coordinate=ulp_g,
              hess_vartheta_eig_min=float(ev_[0]), hess_vartheta_eig_max=float(ev_[-1]),
              terminal_grad_inf=[a['grad_inf'] for a in o['attempts']], terminal_decrement=[a['decrement'] for a in o['attempts']])
    a1['e'] = re
    for k, v in re.items(): P(f"A3-1e {k} {v!r}")
    a1['pass'] = ra['pass'] and rb_['pass'] and rc['pass']
    a1['pass_supp_reading'] = ra['pass'] and rb_['pass_supp'] and rc['pass']
    P(f"A3-1 estimator check (a, b, c as specified; frozen acceptance rule): {'PASS' if a1['pass'] else 'FAIL'} "
      f"(a {ra['pass']}, b {rb_['pass']}, c {rc['pass']}; supporting d {rd['pass']}); under the supplementary rule b would read "
      f"{rb_['pass_supp']}")
    out['A3_1'] = a1

    # ---------------- A3-3 isolation ---------------------------------------------------------------------------------------
    sc_ = iso.static_check('design_r8'); dc = iso.dynamic_check('design_r8')
    sc_e = iso.static_check('estimator_r8', allowed=iso.ALLOWED+('estimator_r8', 'tables_r8'))
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
    a3 = dict(static=sc_, dynamic=dc, static_estimator=sc_e, guarded_holders=names, positive_control=pos, positive_control_pass=pos_ok,
              positive_control_logged=logged,
              restored_after_guard=restored, guarded_design_status=dg['status'], guarded_design_access_log=list(iso.ACCESS_LOG),
              guarded_design_EN_hat=dg.get('EN_hat'), guarded_design_wall_s=t_g)
    a3['pass'] = sc_['passed'] and dc['passed'] and pos_ok and restored and guarded_ok
    for k in ('local', 'third_party', 'outside_allowlist', 'bad_third_party', 'forbidden_names', 'passed'): P(f"A3-3 static design_r8 {k} {sc_[k]!r}")
    for k, v in dc.items(): P(f"A3-3 dynamic {k} {v!r}")
    P(f"A3-3 static estimator_r8 (supporting) local {sc_e['local']} forbidden {sc_e['forbidden_names']} passed {sc_e['passed']}")
    P(f"A3-3 sentinel holders {names}")
    P(f"A3-3 sentinel positive control {pos}; holders logged {logged} -> {'PASS' if pos_ok else 'FAIL'}; originals restored {restored}")
    P(f"A3-3 guarded design at theta_0 + 0.3 sigma e1: status {dg['status']}, sentinel accesses {len(iso.ACCESS_LOG)}, E_hat[N] {dg.get('EN_hat')!r}")
    P(f"A3-3 isolation: {'PASS' if a3['pass'] else 'FAIL'}")
    out['A3_3'] = a3

    # ---------------- A3-4 assertions --------------------------------------------------------------------------------------
    a4 = dict(cons); a4['pass'] = cons['mass_dev_max'] <= 1e-10 and cons['min_m'] >= 0 and cons['assert_fail'] == 0
    for k, v in a4.items(): P(f"A3-4 {k} {v!r}")
    out['A3_4'] = a4
    out['all_required_pass'] = bool(a1['pass'] and a2['pass'] and a3['pass'] and a4['pass'])
    out['supporting_pass'] = bool(dec['pass'] and vb['pass'] and phr['pass'] and sim['pass'] and rd['pass'])
    P(f"A3 SUMMARY: check 1 {'PASS' if a1['pass'] else 'FAIL'}, check 2 {'PASS' if a2['pass'] else 'FAIL'}, check 3 "
      f"{'PASS' if a3['pass'] else 'FAIL'}, check 4 {'PASS' if a4['pass'] else 'FAIL'}; supporting regressions "
      f"{'PASS' if out['supporting_pass'] else 'FAIL'}")
    json.dump(jl(out), open(os.path.join(RES, 'partA_checks.json'), 'w'), indent=1)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--threads', type=int, default=4); a = ap.parse_args()
    sys.stdout = Tee(os.path.join(LOGS, 'partA_checks.log'))
    t0 = time.perf_counter(); o = main(a.threads); print(f"=== checks_r8.py done in {time.perf_counter()-t0:.1f} s")
