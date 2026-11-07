"""Round 10 driver: inputs and their reproduction, the bound UB(B) and V_1 at the two budgets, the KKT cross-check, checks
1-6, outputs (spec reports_CN/plan_round10_logdet_upper_bound_EN.md, frozen 2026-11-08; execution prompt
reports_CN/prompt_round10_claude_code_EN.md §1-§5). Rounds 4-8b are imported read-only; nothing there is changed.

  §1  inputs: K = 5000, sigma = Delta*/20, Hk from round4/evaluator_window.design_grid('U'); every archived input is read at full
      precision AND recomputed with the frozen evaluator (relative error <= 1e-9, else stop):
        U/SoD   evaluator_window.evaluate_table('U', ConstTable(480, 480))       vs mc.json["U/SoD"]{EN_eval, logdet_eval}
        U/W     part_c.table_fields('U/W') (forward of the committed table)      vs mc.json["U/W"]{EN_eval, logdet_eval}
        T4      round7/results/table_T4.npz through evaluate_table (sigma/24)    vs scan_U_W_m020.json runs[1]{EN, logdet}
        SoD@T4  round7 evaluator_refine.evaluate(SodTable(M), 5, theta_0)        vs partD_match.json nominal.T4{EN_SoD, Q}
      full-data log det(sigma^-2 sum h_k h_k^T) is computed and recorded (reference of check 2).
  §3  per budget B in {mc.json U/SoD EN_eval, scan runs[1] EN}: UB (c_max), V_1 (c_one), KKT water-filling (DECISION 10-1),
      gap interval against I_table (nominal table / T4), D-efficiency lower bound exp(-g_U/p), p = 9.
  §4  checks 1-6; §5 outputs results/round10.json, q_opt_B316.npy, q_opt_B263.npy, q_table_nominal.npy, fig_send_profile.png.

Usage (from this directory):  python3 run_round10.py      -> logs/run_round10.log, results/*
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, platform
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R4 = os.path.join(EV, 'round4'); R6 = os.path.join(EV, 'round6'); R7 = os.path.join(EV, 'round7')
for _p in (R7, R4, HERE): sys.path.insert(0, _p)
import numpy as np
import scipy
from scipy.stats import norm, spearmanr
import ub

LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
for _d in (LOGS, RES): os.makedirs(_d, exist_ok=True)
REL_TOL = 1e-9                      # §1 reproduction tolerance
TOL_GAP = 0.01                      # plan §2 stopping rule (nat)
KKT_TOL = 0.01                      # §3 item 3 pass criterion (nat)
S_SYM_REF, S_ONE_REF, S_TOL = 1636.15048, 1346.81561, 1e-3      # check 3 (reviewer's values)
FD_REL, FD_N, FD_TOL, SEED = 1e-3, 20, 1e-6, 20261108           # check 4 (DECISION 10-7)
LEMMA_Q = (0.001, 0.01, 0.063, 0.2, 0.5); LEMMA_N, LEMMA_TOL = 2000, 1e-10   # check 5
F1_TOL = 1e-9                                                    # check 2 (DECISION 10-8)


class Tee:
    def __init__(self, path): self.f = open(path, 'w')
    def __call__(self, m=''): print(m, flush=True); self.f.write(m+'\n'); self.f.flush()

def relerr(x, ref): return abs(float(x)-float(ref))/abs(float(ref))

class Stop(RuntimeError):
    """a reproduction, conservation or check failure: the run stops (no fallback)."""


def main():
    out = Tee(os.path.join(LOGS, 'run_round10.log')); T0 = time.perf_counter()
    out(f"ROUND 10 run {time.strftime('%Y-%m-%d %H:%M:%S')}; python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
    import evaluator_window as ew
    import part_c
    import trigger_dp as td
    import evaluator_refine as er
    import numba
    out(f"numba {numba.__version__}; frozen modules: {ew.__file__}, {part_c.__file__}, {er.__file__}")
    dr = ew.dr; res = dict(inputs={}, reproduction={}, budgets={}, checks={}, decisions_ref='reports_CN/report_round10_CN.md §7')

    # ---------------------------------------------------------------- §1 inputs (archived, full precision) ----------------
    mc = json.load(open(os.path.join(R4, 'results', 'mc.json')))
    scan = json.load(open(os.path.join(R6, 'results', 'scan_U_W_m020.json')))
    pdm = json.load(open(os.path.join(R7, 'results', 'partD_match.json')))
    A = dict(B_nom=mc['U/SoD']['EN_eval'], I_SoD=mc['U/SoD']['logdet_eval'], EN_W=mc['U/W']['EN_eval'], I_W=mc['U/W']['logdet_eval'],
             B_T4=scan['runs'][1]['EN'], I_T4=scan['runs'][1]['logdet'], kappa_T4=scan['runs'][1]['lam_over_gbar'],
             I_SoD_T4=pdm['nominal']['T4']['Q'], EN_SoD_T4=pdm['nominal']['T4']['EN_SoD'], M_T4=pdm['nominal']['T4']['M'])
    src = dict(B_nom='round4/results/mc.json["U/SoD"]["EN_eval"]', I_SoD='round4/results/mc.json["U/SoD"]["logdet_eval"]',
               EN_W='round4/results/mc.json["U/W"]["EN_eval"]', I_W='round4/results/mc.json["U/W"]["logdet_eval"]',
               B_T4='round6/results/scan_U_W_m020.json["runs"][1]["EN"]', I_T4='round6/results/scan_U_W_m020.json["runs"][1]["logdet"]',
               kappa_T4='round6/results/scan_U_W_m020.json["runs"][1]["lam_over_gbar"]',
               I_SoD_T4='round7/results/partD_match.json["nominal"]["T4"]["Q"]', EN_SoD_T4='round7/results/partD_match.json["nominal"]["T4"]["EN_SoD"]',
               M_T4='round7/results/partD_match.json["nominal"]["T4"]["M"]')
    for k, v in A.items(): out(f"INPUT {k} = {v!r}   [{src[k]}]")
    assert scan['runs'][1]['run'] == 2 and abs(A['kappa_T4']-4.0) < 1e-12
    res['inputs'] = {k: dict(value=A[k], source=src[k]) for k in A}

    xk, Hk, z, dz, a, b = ew.design_grid('U'); sigma = ew.s
    K, p = Hk.shape
    assert K == 5000 == ew.K and sigma == ew.D/20 and p == 9, (K, p, sigma)
    tk = np.asarray(dr.t, float); assert tk.shape == (K,)
    out(f"SCENARIO K = {K}, p = {p}, sigma = Delta*/20 = {sigma!r}, Delta* = {ew.D!r}; Hk from round4 evaluator_window.design_grid('U'); "
        f"t_k = k dt, dt = {float(tk[1]-tk[0])!r} s")

    # ---------------------------------------------------------------- §1 reproduction with the frozen evaluator ----------
    rep = {}
    def chk(name, val, ref):
        e = relerr(val, ref); ok = e <= REL_TOL
        rep[name] = dict(recomputed=float(val), archived=float(ref), rel_err=e, pass_=bool(ok))
        out(f"REPRO {name}: recomputed {float(val)!r} archived {float(ref)!r} rel err {e:.3e} -> {'PASS' if ok else 'FAIL'}")
        return ok
    t0 = time.perf_counter()
    rec_s, J_s, r_s = ew.evaluate_table('U', ew.ConstTable(ew.SOD_M, ew.SOD_M))
    ok = chk('U/SoD EN (= B_nom)', rec_s['EN'], A['B_nom']) & chk('U/SoD logdet', rec_s['logdet'], A['I_SoD'])
    J_sod_arch = np.load(os.path.join(R4, 'results', 'J_U_SoD.npy'))
    eJ = float(np.max(np.abs(J_s-J_sod_arch))/np.max(np.abs(J_sod_arch)))
    out(f"REPRO U/SoD J vs round4/results/J_U_SoD.npy: max rel {eJ:.3e}; path {rec_s['path']}; {time.perf_counter()-t0:.1f} s")
    rep['U/SoD J vs J_U_SoD.npy'] = dict(max_rel=eJ, pass_=bool(eJ <= REL_TOL)); ok &= eJ <= REL_TOL

    t0 = time.perf_counter()
    des, tab, fw = part_c.table_fields('U/W')
    assert np.array_equal(des.Hk, Hk) and np.array_equal(des.xk, xk)
    J_w = (Hk*fw['W'][:, None]).T@Hk/sigma**2; ld_w = float(np.linalg.slogdet(J_w)[1])
    ok &= chk('U/W EN', fw['EN'], A['EN_W']) & chk('U/W logdet', ld_w, A['I_W'])
    Q = np.asarray(fw['Q'], float); q_table = Q[1:].copy()
    eQ = relerr(1.0+q_table.sum(), fw['EN'])
    out(f"REPRO U/W per-step transmission probability fw['Q'][1:]: 1 + sum = {float(1.0+q_table.sum())!r} vs fw['EN'] {float(fw['EN'])!r} (rel {eQ:.1e}); "
        f"Q[0] = {float(Q[0])!r}; min {float(q_table.min())!r} max {float(q_table.max())!r}; {time.perf_counter()-t0:.1f} s")
    rep['U/W 1+sum Q[1:] vs EN'] = dict(rel_err=eQ, pass_=bool(eQ <= 1e-12)); ok &= eQ <= 1e-12
    del fw['m_window']

    t0 = time.perf_counter()
    with np.load(os.path.join(R7, 'results', 'table_T4.npz')) as f4:
        codes4, st4, wd4, run4 = f4['codes'], f4['start'], f4['width'], int(f4['run'])
    assert run4 == 2 and np.array_equal(st4, des.a[1:]) and np.array_equal(wd4, des.w[1:])
    rec4, J4, r4 = ew.evaluate_table('U', des.table(codes4))
    ok &= chk('T4 EN (= B_T4)', rec4['EN'], A['B_T4']) & chk('T4 logdet', rec4['logdet'], A['I_T4'])
    out(f"REPRO T4 path {rec4['path']}; {time.perf_counter()-t0:.1f} s")

    t0 = time.perf_counter()
    er.warmup(); pts = {q_['pid']: q_ for q_ in er.points27()}
    recm, Jm, rm = er.evaluate(er.SodTable(int(A['M_T4'])), 5, theta=pts['theta_0']['theta'])
    ok &= chk('SoD@T4 EN (M = %d, sigma/120)' % A['M_T4'], recm['EN'], A['EN_SoD_T4']) & chk('SoD@T4 logdet (Q)', recm['logdet'], A['I_SoD_T4'])
    out(f"REPRO SoD@T4 route {recm['route']}, f = {recm['f']}; E[N] {recm['EN']!r} <= B_T4 {A['B_T4']!r}: {recm['EN'] <= A['B_T4']}; "
        f"{time.perf_counter()-t0:.1f} s")

    J_full = Hk.T@Hk/sigma**2; ld_full = float(np.linalg.slogdet(J_full)[1])
    out(f"FULL-DATA log det(sigma^-2 sum_k h_k h_k^T) = {ld_full!r} (not archived; computed here, reference of check 2)")
    res['reproduction'] = rep; res['inputs']['I_full'] = dict(value=ld_full, source='computed: slogdet(Hk.T@Hk/sigma^2)')
    if not ok: raise Stop("reproduction of the archived inputs failed (relative error > 1e-9): stop")
    out("REPRO all archived inputs reproduced within 1e-9 -> proceed")

    # ---------------------------------------------------------------- checks 2-5 (independent of the solve) ---------------
    PS = ub.Problem(Hk, sigma, ub.c_max, ub.dc_max); PO = ub.Problem(Hk, sigma, ub.c_one, ub.dc_one)
    C = res['checks']
    f1s = PS.f(np.ones(K-1)); f1o = PO.f(np.ones(K-1))
    C['2_full_data'] = dict(f_sym_q1=f1s, f_one_q1=f1o, full_logdet=ld_full, diff_sym=f1s-ld_full, diff_one=f1o-ld_full,
                            tol=F1_TOL, pass_=bool(abs(f1s-ld_full) <= F1_TOL and abs(f1o-ld_full) <= F1_TOL))
    out(f"CHECK 2 f(q = 1): c_max {f1s!r}, c_one {f1o!r}; full-data {ld_full!r}; diffs {f1s-ld_full:.2e}, {f1o-ld_full:.2e} "
        f"(|.| <= {F1_TOL}) -> {'PASS' if C['2_full_data']['pass_'] else 'FAIL'}")

    def scalar(Bv, c, dc):
        n = K-1; Bm1 = Bv-1; q = np.full(n, Bm1/n)
        S = 1.0+float(c(q).sum()); g = dc(q); U, _ = ub.u_cert(S, g, q, Bm1)
        return S, U-S, float(q[0])
    Ss, gs, qu = scalar(A['B_nom'], ub.c_max, ub.dc_max); So, go, _ = scalar(A['B_nom'], ub.c_one, ub.dc_one)
    Ss_r, _, _ = scalar(316.892, ub.c_max, ub.dc_max); So_r, _, _ = scalar(316.892, ub.c_one, ub.dc_one)
    C['3_scalar'] = dict(B=A['B_nom'], q_uniform=qu, S_sym=Ss, S_one=So, S_sym_ref=S_SYM_REF, S_one_ref=S_ONE_REF,
                         diff_sym=Ss-S_SYM_REF, diff_one=So-S_ONE_REF, cert_gap_sym=gs, cert_gap_one=go, tol=S_TOL,
                         S_sym_at_rounded_B_316_892=Ss_r, S_one_at_rounded_B_316_892=So_r,
                         convention='S = 1 + sum_{k>=2} c(q_k) (h_1 always transmitted, weight 1; H[k] -> 1 in f), maximised at uniform q',
                         pass_=bool(abs(Ss-S_SYM_REF) <= S_TOL and abs(So-S_ONE_REF) <= S_TOL))
    out(f"CHECK 3 scalar (S = 1 + sum c(q_k), optimum uniform q = (B-1)/(K-1) = {qu!r}, certificate gap {gs:.1e} / {go:.1e}): "
        f"S_sym {Ss!r} (ref {S_SYM_REF}, diff {Ss-S_SYM_REF:+.5f}), S_one {So!r} (ref {S_ONE_REF}, diff {So-S_ONE_REF:+.5f}); "
        f"tol {S_TOL} -> {'PASS' if C['3_scalar']['pass_'] else 'FAIL'}")
    out(f"CHECK 3 diagnostic: at the rounded B = 316.892, S_sym {Ss_r!r}, S_one {So_r!r}")

    def grad_check(P, q, rng, tag):
        f, g, _ = P.fg(q); idx = rng.choice(K-1, FD_N, replace=False); rows = []
        for k in idx:
            h = FD_REL*q[k]; qp = q.copy(); qm = q.copy(); qp[k] += h; qm[k] -= h
            fd = (P.f(qp)-P.f(qm))/(2*h); rows.append((int(k), float(g[k]), float(fd), abs(fd-g[k])/abs(g[k])))
        mx = max(r_[3] for r_ in rows)
        out(f"CHECK 4 {tag}: max rel err over {FD_N} components {mx:.3e} (h_k = {FD_REL} q_k, central difference)")
        return dict(max_rel=mx, components=[dict(k=r_[0]+2, analytic=r_[1], fd=r_[2], rel=r_[3]) for r_ in rows])
    rng = np.random.default_rng(SEED)
    q0 = np.full(K-1, (A['B_nom']-1)/(K-1))
    qr = q0*np.exp(0.5*rng.standard_normal(K-1)); qr *= (A['B_nom']-1)/qr.sum()
    g4 = {'sym@q0': grad_check(PS, q0, rng, 'c_max at q0 = (B-1)/(K-1)'), 'one@q0': grad_check(PO, q0, rng, 'c_one at q0'),
          'sym@qrand': grad_check(PS, qr, rng, 'c_max at a random interior point'), 'one@qrand': grad_check(PO, qr, rng, 'c_one at the random point')}

    lem = {}
    for qv in LEMMA_Q:
        u = np.arange(1, LEMMA_N)/LEMMA_N                          # u = Phi(alpha)/q in (0, 1): 1999 points (u = 1/2 included)
        al = norm.ppf(u*qv); be = norm.isf(qv*(1-u))                # Phi(beta) - Phi(alpha) = 1 - q along the scan
        al_s = -norm.isf(qv/2); be_s = norm.isf(qv-norm.cdf(al_s))  # the theoretical symmetric point, explicitly
        al = np.append(al, al_s); be = np.append(be, be_s)
        cev, cno = dr.cfun(al, be); c = cev+cno                     # frozen mc83.cfun
        cm = float(ub.c_max(qv)); exc = float(c.max()-cm); att = float(abs(c[-1]-cm))
        stay = norm.cdf(be)-norm.cdf(al); dst = float(np.max(np.abs(stay-(1-qv))))
        ok5 = exc <= LEMMA_TOL and att <= LEMMA_TOL
        lem[str(qv)] = dict(n_points=int(al.size), alpha_sym=float(al_s), beta_sym=float(be_s), c_max=cm, c_at_sym=float(c[-1]),
                            max_scan=float(c.max()), argmax_u=float(np.append(u, 0.5)[int(np.argmax(c))]), max_minus_cmax=exc,
                            attain_err=att, min_scan=float(c.min()), max_stay_prob_err=dst, pass_=bool(ok5))
        out(f"CHECK 5 q = {qv}: {al.size} alpha points incl. alpha_sym = {float(al_s)!r}; max c - c_max = {exc:.2e} (<= {LEMMA_TOL}); "
            f"|c(sym) - c_max| = {att:.2e}; c range [{c.min():.6f}, {c.max():.6f}], c_max {cm!r}; constraint err {dst:.1e} -> {'PASS' if ok5 else 'FAIL'}")
    qf = np.logspace(-4, np.log10(0.99), 200); worst = -np.inf
    for qv in qf:                                                    # plan §4.5 "fine grid of q" (DECISION 10-9; descriptive)
        u = np.arange(1, LEMMA_N)/LEMMA_N; al = norm.ppf(u*qv); be = norm.isf(qv*(1-u))
        cev, cno = dr.cfun(al, be); worst = max(worst, float((cev+cno).max()-ub.c_max(qv)))
    out(f"CHECK 5 fine q grid (200 log-spaced q in [1e-4, 0.99], 1999 alpha each): worst max c - c_max = {worst:.2e}")
    C['5_lemmaA'] = dict(per_q=lem, fine_grid_worst_excess=worst, fine_grid_pass=bool(worst <= LEMMA_TOL),
                         pass_=bool(all(v['pass_'] for v in lem.values())))

    # ---------------------------------------------------------------- §3 solves --------------------------------------------
    budgets = [('B316', A['B_nom'], A['I_W'], 'nominal table U/W', src['I_W'], 'q_opt_B316.npy'),
               ('B263', A['B_T4'], A['I_T4'], 'T4', src['I_T4'], 'q_opt_B263.npy')]
    sol = {}
    for tag, Bv, I_tab, tname, tsrc, qfile in budgets:
        Bm1 = Bv-1; out(f"SOLVE {tag}: B = {Bv!r}, B - 1 = {Bm1!r}; I_table = {I_tab!r} ({tname}, {tsrc})")
        PS = ub.Problem(Hk, sigma, ub.c_max, ub.dc_max); out(" UB (c_max), projected gradient ascent:")
        rS = ub.solve(PS, Bm1, tol=TOL_GAP, log=out)
        PO = ub.Problem(Hk, sigma, ub.c_one, ub.dc_one); out(" V1 (c_one), projected gradient ascent:")
        rO = ub.solve(PO, Bm1, tol=TOL_GAP, log=out)
        out(" KKT water-filling (c_max):"); kS = ub.kkt_fixed_point(ub.Problem(Hk, sigma, ub.c_max, ub.dc_max), Bm1, ub.INV_DC['sym'], log=out)
        out(" KKT water-filling (c_one):"); kO = ub.kkt_fixed_point(ub.Problem(Hk, sigma, ub.c_one, ub.dc_one), Bm1, ub.INV_DC['one'], log=out)
        gL = rS['f']-I_tab; gU = rS['u_cert']-I_tab
        d = dict(B=Bv, Bm1=Bm1, I_table=I_tab, I_table_name=tname, I_table_source=tsrc,
                 f=rS['f'], u_cert=rS['u_cert'], gap=rS['gap'], g_L=gL, g_U=gU, deff_lower=float(np.exp(-gU/p)),
                 lambda_hat=rS['lambda_hat'], n_iter=rS['n_iter'], sum_q=rS['sum_q'], q_min=float(rS['q'].min()), q_max=float(rS['q'].max()),
                 n_f_evals=rS['n_f'], wall_s=rS['wall_s'], n_damped=rS['n_damped'], n_backtrack=rS['n_backtrack'], n_t_shrink=rS['n_t_shrink'],
                 V1_n_damped=rO['n_damped'], V1_n_backtrack=rO['n_backtrack'],
                 V1_f=rO['f'], V1_u_cert=rO['u_cert'], V1_gap=rO['gap'], V1_n_iter=rO['n_iter'], V1_lambda_hat=rO['lambda_hat'],
                 kkt_f=kS['f'], kkt_u_cert=kS['u_cert'], kkt_gap=kS['gap'], kkt_lambda=kS['lam'], kkt_n_iter=kS['n_iter'],
                 kkt_max_dq=kS['dq'], kkt_f_minus_f=kS['f']-rS['f'], kkt_pass=bool(abs(kS['f']-rS['f']) <= KKT_TOL),
                 kkt_max_abs_q_diff=float(np.max(np.abs(kS['q']-rS['q']))),
                 V1_kkt_f=kO['f'], V1_kkt_u_cert=kO['u_cert'], V1_kkt_gap=kO['gap'], V1_kkt_f_minus_f=kO['f']-rO['f'],
                 tight_g_L=kS['f']-I_tab, tight_g_U=kS['u_cert']-I_tab, tight_deff_lower=float(np.exp(-(kS['u_cert']-I_tab)/p)),
                 history=[dict(it=h[0], f=h[1], u_cert=h[2], gap=h[3], t=h[4]) for h in rS['history']],
                 V1_history=[dict(it=h[0], f=h[1], u_cert=h[2], gap=h[3], t=h[4]) for h in rO['history']])
        d['V1_minus_I_table'] = rO['f']-I_tab
        sol[tag] = dict(rS=rS, rO=rO, kS=kS, kO=kO); res['budgets'][tag] = d
        np.save(os.path.join(RES, qfile), rS['q'])
        out(f" {tag}: f(q_hat) {rS['f']!r}  U_cert {rS['u_cert']!r}  gap {rS['gap']:.3e}  iterations {rS['n_iter']}  lambda_hat {rS['lambda_hat']!r}"
            f"  damped steps {rS['n_damped']}, backtracks {rS['n_backtrack']}, t shrinks {rS['n_t_shrink']}; V1 damped {rO['n_damped']}, backtracks {rO['n_backtrack']}")
        out(f" {tag}: g_L {gL!r}  g_U {gU!r}  D-eff lower bound exp(-g_U/{p}) {float(np.exp(-gU/p))!r}")
        out(f" {tag}: V1 f {rO['f']!r}  U_cert {rO['u_cert']!r}  gap {rO['gap']:.3e}  iterations {rO['n_iter']}")
        out(f" {tag}: KKT f {kS['f']!r} (f_KKT - f_PGA = {kS['f']-rS['f']:+.3e}; |.| <= {KKT_TOL}: {d['kkt_pass']}), its U_cert {kS['u_cert']!r} "
            f"(gap {kS['gap']:.1e}), lambda {kS['lam']!r}, {kS['n_iter']} fixed-point iterations")
    np.save(os.path.join(RES, 'q_table_nominal.npy'), q_table)

    # checks 4 at q_hat, 1, 6 --------------------------------------------------------------------------------------------
    rng2 = np.random.default_rng(SEED+1)
    g4['sym@qhat_B316'] = grad_check(ub.Problem(Hk, sigma, ub.c_max, ub.dc_max), sol['B316']['rS']['q'], rng2, 'c_max at q_hat (B316)')
    g4['one@qhat_B316'] = grad_check(ub.Problem(Hk, sigma, ub.c_one, ub.dc_one), sol['B316']['rO']['q'], rng2, 'c_one at q_hat_V1 (B316)')
    C['4_gradient'] = dict(points=g4, h_rel=FD_REL, tol=FD_TOL, seed=SEED, pass_=bool(all(v['max_rel'] <= FD_TOL for v in g4.values())))
    out(f"CHECK 4 -> {'PASS' if C['4_gradient']['pass_'] else 'FAIL'} (worst {max(v['max_rel'] for v in g4.values()):.3e})")

    U316 = res['budgets']['B316']['u_cert']; U263 = res['budgets']['B263']['u_cert']
    c1 = dict(U_cert_B316=U316, U_cert_B263=U263,
              B316_vs_UW=dict(policy=A['I_W'], EN=A['EN_W'], margin=U316-A['I_W'], pass_=bool(U316 >= A['I_W'])),
              B316_vs_USoD=dict(policy=A['I_SoD'], EN=A['B_nom'], margin=U316-A['I_SoD'], pass_=bool(U316 >= A['I_SoD'])),
              B263_vs_T4=dict(policy=A['I_T4'], EN=A['B_T4'], margin=U263-A['I_T4'], pass_=bool(U263 >= A['I_T4'])),
              B263_vs_SoD_M581=dict(policy=A['I_SoD_T4'], EN=A['EN_SoD_T4'], margin=U263-A['I_SoD_T4'], pass_=bool(U263 >= A['I_SoD_T4'])))
    c1['pass_'] = all(v['pass_'] for k, v in c1.items() if isinstance(v, dict))
    C['1_budget_matching'] = c1
    for k, v in c1.items():
        if isinstance(v, dict): out(f"CHECK 1 {k}: U_cert - I_policy = {v['margin']!r} (policy E[N] {v['EN']!r}) -> {'PASS' if v['pass_'] else 'FAIL'}")
    C['6_gap'] = dict(B316=res['budgets']['B316']['gap'], B263=res['budgets']['B263']['gap'], tol=TOL_GAP,
                      n_iter=dict(B316=res['budgets']['B316']['n_iter'], B263=res['budgets']['B263']['n_iter']),
                      pass_=bool(res['budgets']['B316']['gap'] <= TOL_GAP and res['budgets']['B263']['gap'] <= TOL_GAP))
    C['kkt_crosscheck'] = dict(B316=res['budgets']['B316']['kkt_f_minus_f'], B263=res['budgets']['B263']['kkt_f_minus_f'], tol=KKT_TOL,
                               pass_=bool(res['budgets']['B316']['kkt_pass'] and res['budgets']['B263']['kkt_pass']))
    out(f"CHECK 6 gap <= {TOL_GAP}: B316 {C['6_gap']['B316']:.3e}, B263 {C['6_gap']['B263']:.3e} -> {'PASS' if C['6_gap']['pass_'] else 'FAIL'}")

    # ---------------------------------------------------------------- profile statistics (descriptive) --------------------
    qh = sol['B316']['rS']['q']; qh2 = sol['B263']['rS']['q']
    L = np.linalg.cholesky(J_sod_arch); Y = np.linalg.solve(L, Hk[1:].T); gam_sod = np.sum(Y*Y, axis=0)/sigma**2
    prof = dict(spearman_qhat_qtable=float(spearmanr(qh, q_table).statistic),
                spearman_qhat263_qtable=float(spearmanr(qh2, q_table).statistic),
                spearman_qhat_gamma_sod=float(spearmanr(qh, gam_sod).statistic),
                spearman_qtable_gamma_sod=float(spearmanr(q_table, gam_sod).statistic),
                qhat_min=float(qh.min()), qhat_max=float(qh.max()), qhat_ratio=float(qh.max()/qh.min()), qhat_cv=float(qh.std()/qh.mean()),
                qtable_min=float(q_table.min()), qtable_max=float(q_table.max()), qtable_cv=float(q_table.std()/q_table.mean()),
                qhat_sum=float(qh.sum()), qtable_sum=float(q_table.sum()), q_uniform=float((A['B_nom']-1)/(K-1)),
                gamma_sod_source='round4/results/J_U_SoD.npy (sigma^-2 h_k^T J_SoD^-1 h_k, k = 2..K)')
    res['profile'] = prof
    for k, v in prof.items(): out(f"PROFILE {k} = {v!r}")
    res['derived'] = dict(delta_I_UB=U316-A['I_SoD'], delta_I_table=A['I_W']-A['I_SoD'], gain_fraction_table_over_UB=(A['I_W']-A['I_SoD'])/(U316-A['I_SoD']),
                          V1_minus_I_SoD=res['budgets']['B316']['V1_f']-A['I_SoD'], V1_ucert_minus_I_SoD=res['budgets']['B316']['V1_u_cert']-A['I_SoD'],
                          UB_minus_V1_B316=res['budgets']['B316']['f']-res['budgets']['B316']['V1_f'],
                          delta_I_UB_B263_vs_SoD_M581=U263-A['I_SoD_T4'], delta_I_T4_vs_SoD_M581=A['I_T4']-A['I_SoD_T4'],
                          full_minus_UB316=ld_full-U316)
    for k, v in res['derived'].items(): out(f"DERIVED {k} = {v!r}")

    # ---------------------------------------------------------------- figure ------------------------------------------------
    fig_path = os.path.join(RES, 'fig_send_profile.png')
    make_figure(tk, qh, q_table, prof, A, fig_path)
    out(f"FIGURE {os.path.relpath(fig_path, HERE)}")

    # ---------------------------------------------------------------- summary tables and json --------------------------------
    allpass = all(C[k]['pass_'] for k in ('1_budget_matching', '2_full_data', '3_scalar', '4_gradient', '5_lemmaA', '6_gap')) and C['kkt_crosscheck']['pass_']
    res['all_checks_pass'] = bool(allpass); res['wall_s'] = time.perf_counter()-T0
    out(""); out("TABLE 1  results per budget (nat)")
    out(f"{'budget B':>20} {'I_table':>12} {'f(q_hat)':>12} {'U_cert':>12} {'gap':>9} {'g_L':>8} {'g_U':>8} {'Deff>=':>7} {'V1 f':>12} {'V1 U_cert':>12} {'KKT f':>12}")
    for tag in ('B316', 'B263'):
        d = res['budgets'][tag]
        out(f"{d['B']:>20.13f} {d['I_table']:>12.6f} {d['f']:>12.6f} {d['u_cert']:>12.6f} {d['gap']:>9.2e} {d['g_L']:>8.4f} {d['g_U']:>8.4f} "
            f"{100*d['deff_lower']:>6.2f}% {d['V1_f']:>12.6f} {d['V1_u_cert']:>12.6f} {d['kkt_f']:>12.6f}")
    out(""); out("TABLE 2  checks")
    for k in ('1_budget_matching', '2_full_data', '3_scalar', '4_gradient', '5_lemmaA', '6_gap', 'kkt_crosscheck'):
        out(f"  {k:<20} {'PASS' if C[k]['pass_'] else 'FAIL'}")
    out(f"ALL CHECKS {'PASS' if allpass else 'FAIL'}; wall {res['wall_s']:.1f} s")
    json.dump(res, open(os.path.join(RES, 'round10.json'), 'w'), indent=1)
    out(f"WROTE results/round10.json, results/q_opt_B316.npy, results/q_opt_B263.npy, results/q_table_nominal.npy, results/fig_send_profile.png")
    if not allpass: raise Stop("a round-10 check failed: see the log; the report records it")


def make_figure(tk, qh, q_table, prof, A, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'legend.fontsize': 9, 'font.family': 'DejaVu Sans'})
    BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
    t = tk[1:]; qu = prof['q_uniform']
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 6.4), gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.42))
    fig.patch.set_facecolor('#fcfcfb')
    zoom = (2.0, 2.5)
    for ax, (lo, hi), lw, ttl in ((axs[0], (t[0], t[-1]), 0.8, '(a) whole experiment, 0-5 s'),
                                  (axs[1], zoom, 1.6, f'(b) zoom {zoom[0]:.1f}-{zoom[1]:.1f} s')):
        m = (t >= lo) & (t <= hi)
        ax.set_facecolor('#fcfcfb')
        ax.plot(t[m], q_table[m], color=ORANGE, lw=lw, solid_joinstyle='round', solid_capstyle='round',
                label='nominal table U/W: E[q_k] from the evaluator (fw[\'Q\'][1:])')
        ax.plot(t[m], qh[m], color=BLUE, lw=lw, solid_joinstyle='round', solid_capstyle='round',
                label='bound optimum $\\hat q_k$ (symmetric window, B = 316.892)')
        ax.axhline(qu, color='#8a8984', lw=0.8, zorder=1, label=f'uniform allocation (B-1)/(K-1) = {qu:.4f}')
        ax.set_xlim(lo, hi); ax.set_ylim(0, 0.16)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'): ax.spines[sp].set_color('#b5b4ae')
        ax.tick_params(colors=MUTED); ax.set_title(ttl, loc='left', color=INK)
        ax.set_xlabel('time t_k (s)', color=INK); ax.set_ylabel('per-step transmission probability', color=INK)
    axs[0].legend(loc='upper left', frameon=False, ncol=1, bbox_to_anchor=(0.0, 1.03), handlelength=1.6, labelcolor=INK)
    cap = (f"What to look at: where the two profiles rise and fall together and where they part. Spearman rank correlation over "
           f"k = 2..K: {prof['spearman_qhat_qtable']:.3f}. Range of $\\hat q_k$ [{prof['qhat_min']:.3f}, {prof['qhat_max']:.3f}], "
           f"of the table [{prof['qtable_min']:.3f}, {prof['qtable_max']:.3f}]; both sum to about B - 1 over the 4999 steps.")
    import textwrap
    fig.text(0.07, 0.012, textwrap.fill(cap, 100), fontsize=9, color=MUTED, ha='left', va='bottom')
    fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.17)
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    main()
