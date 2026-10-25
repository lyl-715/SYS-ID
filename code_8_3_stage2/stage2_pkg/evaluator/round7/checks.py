"""Round 7, Part A2-A4: checks of the refined evaluator (spec reports_CN/plan_round7_CN.md v3.1, frozen; §3 items 1 and 3).

A-CELL  the cell lookup on the production grids: f = 1, 3, 5 (and the half-step-shifted sigma/48 grid of Part B): smallest
        distance of a node to a cell boundary; for odd f the physical lookup equals the integer rule (j + (f - 1)/2) // f;
        on the unshifted sigma/48 grid the boundary assertion must fire (odd nodes lie on the boundaries).
A2 (a)  (M^-, M^+) = (480 f, 480 f) -> reuse path: the dispatched result equals, bit for bit (E[N], S, 81 J elements; also
        E[N]_P, log det, the 5000 w_k), both evaluator_fast.evaluate_fast('U', 24 f) called directly and the accepted
        density_recursion_v1.evaluate('B', 'U', 24 f) (full grid), uniform input at theta_0, f = 1, 3, 5.
   (b)  every other constant window (f M, f M) goes to the general path: dispatch and refined rows checked for
        M in {120, 479, 481, 602, 960} at f = 1, 3, 5; and at f = 1 the general path with M = 602 at theta_0 equals round 6's
        equal-send archive (round6/results/archive_partC.jsonl, theta_0, M = 602) bit for bit.
   (c)  at M = 480: general path (forced) vs reuse path, f = 1, 3, 5: E[N], S relative <= 1e-12; J elements
        |Delta| <= 1e-12 (|ref| + max|J|).
   old  round 4's 24 regression (evaluator_window.regression_check: U and P5514, dispatch bitwise, general within tolerance)
        and the 48 regression of rounds 4/6 (round4/results/c2.json: SoD at n = 48 via the reuse path, m = 0 table via round 4's
        FineTable, bit for bit; general (960, 960) vs reuse within tolerance).
   f = 1 identity: the new evaluator at f = 1 reproduces round 6's archive at theta_0 for m000, m005, m010, m020 and the SoD
        (E[N], E[N]_P, S, log det, 81 J elements; bit for bit).
A3      small-grid explicit-matrix check at f = 3 and 5: round 4's configuration (coarse G = 200, h = sigma/24, z_0 = -4 sigma,
        K = 30, x_k = 2 sigma sin(2 pi k / 30), tables (a) constant, (b) step-dependent, (c) node-dependent with the frozen
        rng, gamma = 1 and the frozen random gamma) on a fine grid of G = 200 f nodes, z_0 + j h/f. The dense side builds every
        window DIRECTLY from the coarse table's physical coordinates: nearest stored coarse node by brute-force physical
        distance (a unique minimiser asserted), physical half-widths M^+- h, membership and the 1/2 end weights from
        physical offsets -- not through evaluator_refine.cell_index / RefinedTable. Errors (infinity-norm ratios) <= 1e-12
        for m_k at every step, E[N], F (forward), V^c_k, V^q_k at every step, E[N], F (backward).
A4      conservation, non-negativity, support assertions: collected from every evaluation above.

Usage (from this directory): python3 checks.py   -> logs/partA_checks.log, results/partA_checks.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import evaluator_refine as er
ew, ef, dr, rb = er.ew, er.ef, er.dr, er.rb
s, K, B = er.s, er.K, er.B
TOL = 1e-12
A4 = dict(n=0, mass_dev_max=0.0, min_m=np.inf, assert_fail=0, inv_all=True)

def a4(rec):
    A4['n'] += 1; A4['mass_dev_max'] = max(A4['mass_dev_max'], float(rec['mass_dev_max'])); A4['min_m'] = min(A4['min_m'], float(rec['min_m']))
    A4['assert_fail'] += int(rec.get('assert_fail', 0)); A4['inv_all'] &= bool(rec['inv_S'] and rec['inv_J'] and rec['inv_w'])

def relerr(y, yref):
    y = np.asarray(y, float); yref = np.asarray(yref, float)
    return float(np.max(np.abs(y-yref))/max(float(np.max(np.abs(yref))), 1e-300))


def cell_checks(out):
    res = {}
    for f in (1, 3, 5):
        g = er.grid(f); j = np.arange(g['G'])
        ii = (j+(f-1)//2)//f
        res[f'f{f}'] = dict(G=g['G'], h_over_sigma=g['dz']/s, min_gap_cells=g['boundary_gap'], integer_rule_equal=bool(np.array_equal(g['cell'], ii)),
                            expected_gap=0.5-(f-1)/(2*f), cells_first_last=[int(g['cell'][0]), int(g['cell'][-1])])
        out(f"A-CELL f={f} G {g['G']} h = sigma/{24*f}; min distance to a cell boundary {g['boundary_gap']!r} cell widths (expected 1/2 - (f-1)/(2f) = "
            f"{0.5-(f-1)/(2*f)!r}); physical lookup equals the integer rule (j + (f-1)/2) // f for all nodes: {res[f'f{f}']['integer_rule_equal']}")
    g = er.grid(2, shift=True)
    res['f2_shift'] = dict(G=g['G'], min_gap_cells=g['boundary_gap'])
    out(f"A-CELL sigma/48 shifted by h_48/2: G {g['G']}; min distance to a cell boundary {g['boundary_gap']!r} cell widths (expected 0.25)")
    try:
        er.grid(2); fired = False; msg = ''
    except er.BoundaryNode as e:
        fired = True; msg = str(e)
    res['f2_unshifted_assertion_fires'] = fired
    out(f"A-CELL sigma/48 unshifted (odd nodes on the cell boundaries): boundary assertion fires: {fired} ({msg})")
    ok = all(res[f'f{f}']['integer_rule_equal'] and abs(res[f'f{f}']['min_gap_cells']-res[f'f{f}']['expected_gap']) < 1e-9 for f in (1, 3, 5)) and fired \
        and abs(res['f2_shift']['min_gap_cells']-0.25) < 1e-9
    out(f"A-CELL cell lookup on the production grids: {'PASS' if ok else 'FAIL'}")
    res['pass_'] = ok
    return res


def a2(out):
    res = {}; ok = True
    th0 = rb.THETA0.copy(); sod = er.SodTable(480)
    for f in er.FACTORS:
        n = 24*f
        t0 = time.perf_counter(); rd, Jd, rrd = er.evaluate(sod, f, th0); t_d = time.perf_counter()-t0; a4(rd)
        rf, Jf, rrf = ef.evaluate_fast('U', n)
        t0 = time.perf_counter(); rg, Jg, rrg = dr.evaluate('B', 'U', n, 8); t_g = time.perf_counter()-t0
        def bit(r1, J1, w1, r2, J2, w2):
            return dict(EN=r1['EN'] == r2['EN'], S=r1['S'] == r2['S'], J81=bool(np.array_equal(J1, J2)), EN_P=r1['EN_P'] == r2['EN_P'],
                        logdet=r1['logdet'] == r2['logdet'], w5000=bool(np.array_equal(w1, w2)))
        b1 = bit(rd, Jd, rrd['W'], rf, Jf, rrf['W']); b2 = bit(rd, Jd, rrd['W'], rg, Jg, rrg['W'])
        pa = rd['route'] == 'reuse' and all(b1.values()) and all(b2.values())
        res[f'a_f{f}'] = dict(route=rd['route'], vs_evaluate_fast=b1, vs_density_recursion_full=b2, EN=rd['EN'], S=rd['S'], logdet=rd['logdet'],
                              G=int(rd['G']), t_dispatch_s=t_d, t_full_grid_s=t_g, pass_=pa)
        ok &= pa
        out(f"A2a f={f} (480 f, 480 f) = ({480*f}, {480*f}) route {rd['route']}; E[N] {rd['EN']!r} S {rd['S']!r} log det {rd['logdet']!r} G {rd['G']}")
        out(f"A2a f={f} bitwise vs evaluator_fast.evaluate_fast('U', {n}) direct: {b1}")
        out(f"A2a f={f} bitwise vs density_recursion_v1.evaluate('B', 'U', {n}) full grid ({t_g:.1f} s): {b2}")
        out(f"A2a f={f} reuse path bitwise identity: {'PASS' if pa else 'FAIL'}")
        # (c) general path at M = 480 vs reuse
        t0 = time.perf_counter(); rG, JG, rrG = er.evaluate(sod, f, th0, force_general=True); t_G = time.perf_counter()-t0; a4(rG)
        Jm = float(np.max(np.abs(Jd)))
        dc = dict(EN_rel=abs(rG['EN']-rd['EN'])/abs(rd['EN']), S_rel=abs(rG['S']-rd['S'])/abs(rd['S']),
                  J_ratio_max=float(np.max(np.abs(JG-Jd)/(np.abs(Jd)+Jm))), logdet_diff=rG['logdet']-rd['logdet'],
                  EN_P_rel=abs(rG['EN_P']-rd['EN_P'])/abs(rd['EN_P']), w_maxabs=float(np.max(np.abs(rrG['W']-rrd['W']))), route=rG['route'], t_general_s=t_G)
        pc = rG['route'] == 'general' and dc['EN_rel'] <= TOL and dc['S_rel'] <= TOL and dc['J_ratio_max'] <= TOL
        dc['pass_'] = pc; res[f'c_f{f}'] = dc; ok &= pc
        for k, v in dc.items(): out(f"A2c f={f} general (480 f, 480 f) vs reuse {k} {v!r}")
        out(f"A2c f={f} tolerance (E[N], S rel <= 1e-12; |dJ| <= 1e-12 (|ref| + max|J|)): {'PASS' if pc else 'FAIL'}")
    # (b) dispatch of the other constant windows
    pb = True
    for f in er.FACTORS:
        g = er.grid(f); a, b = ew.windows88(er.I24['xbar'], g['z'][0], g['dz'], g['G'])
        for M in (120, 479, 481, 602, 960):
            t = er.SodTable(M); rt = er.RefinedTable(t, f, g['cell'])
            rows_ok = all(bool(np.all(rt.rows(p, int(a[p]), int(b[p]))[0] == f*M) and np.all(rt.rows(p, int(a[p]), int(b[p]))[1] == f*M)) for p in (1, K//2, K-1))
            routed = 'reuse' if (t.is_sod()) else 'general'
            pb &= rows_ok and routed == 'general'
            out(f"A2b f={f} M={M}: dispatch {routed}; refined windows ({f*M}, {f*M}) at steps 2, {K//2+1}, {K}: {rows_ok}")
    A6 = {(r['point'], r['M']): r for r in (json.loads(l) for l in open(os.path.join(er.R6, 'results', 'archive_partC.jsonl')))}
    ref = A6[('theta0|theta_0|+|0.0', 602)]                     # round 6's point id of theta_0
    r6, J6, rr6 = er.evaluate(er.SodTable(602), 1, th0); a4(r6)
    eq = dict(EN=r6['EN'] == ref['EN'], EN_P=r6['EN_P'] == ref['EN_P'], S=r6['S'] == ref['S'], logdet=r6['logdet'] == ref['logdet'], route=r6['route'] == 'general')
    pb &= all(eq.values())
    out(f"A2b f=1 M=602 at theta_0, general path vs round6/results/archive_partC.jsonl (exact): {eq}; E[N] {r6['EN']!r}")
    out(f"A2b constant windows other than (480 f, 480 f) go to the general path: {'PASS' if pb else 'FAIL'}")
    res['b'] = dict(pass_=pb, M602_vs_round6=eq); ok &= pb
    # old 24 / 48 regressions
    ok24, r24 = ew.regression_check(out)
    out(f"A2-old24 round 4 regression_check (U, P5514; dispatch bitwise, general within tolerance): {'PASS' if ok24 else 'FAIL'}")
    c2 = json.load(open(os.path.join(er.R4, 'results', 'c2.json')))
    r48, J48, _ = er.eval_reuse(2, er.I24['xbar']); a4(r48)
    T = rb.load_tables(('U/W',)); ft = ew.FineTable(ew.CodeTable(T['U/W'].codes, T['U/W'].start, T['U/W'].width))
    rw, Jw, rrw = ew.evaluate_table('U', ft, n=48); a4(rw)
    rg48, Jg48, _ = ew.evaluate_table('U', ew.ConstTable(960, 960), n=48); a4(rg48)
    e48 = dict(SoD_EN=r48['EN'] == c2['U/SoD']['EN48'], SoD_S=r48['S'] == c2['U/SoD']['S48'], SoD_logdet=r48['logdet'] == c2['U/SoD']['logdet48'],
               m000_EN=rw['EN'] == c2['U/W']['EN48'], m000_S=rw['S'] == c2['U/W']['S48'], m000_logdet=rw['logdet'] == c2['U/W']['logdet48'])
    gen48 = abs(rg48['EN']-r48['EN'])/r48['EN']
    ok48 = all(e48.values()) and gen48 <= TOL and gen48 == c2['U/SoD']['general_vs_accepted_EN_rel']
    out(f"A2-old48 sigma/48 vs round4/results/c2.json (exact): {e48}; general (960, 960) vs reuse E[N] rel {gen48!r} (c2 {c2['U/SoD']['general_vs_accepted_EN_rel']!r})")
    out(f"A2-old48 regression: {'PASS' if ok48 else 'FAIL'}")
    res['old24'] = dict(pass_=ok24); res['old48'] = dict(exact=e48, general_vs_reuse_EN_rel=gen48, pass_=ok48); ok &= ok24 and ok48
    # f = 1 identity vs the round-6 archive at theta_0
    A = {r['id']: r for r in (json.loads(l) for l in open(os.path.join(er.R6, 'results', 'archive_partA.jsonl')))}
    J6a = np.load(os.path.join(er.R6, 'results', 'J_partA.npz'))
    T6 = er.load_tables(('m000', 'm005', 'm010', 'm020')); p1 = True; rows = {}
    for tid in ('SoD', 'm000', 'm005', 'm010', 'm020'):
        rid = f"U|{tid}|theta0|theta_0|+|0.0"; ref = A[rid]
        rr_, JJ, _ = er.evaluate(sod if tid == 'SoD' else T6[tid], 1, th0); a4(rr_)
        e = dict(EN=rr_['EN'] == ref['EN'], EN_P=rr_['EN_P'] == ref['EN_P'], S=rr_['S'] == ref['S'], logdet=rr_['logdet'] == ref['logdet'],
                 J81=bool(np.array_equal(JJ, J6a[rid])))
        rows[tid] = e; p1 &= all(e.values())
        out(f"A2-f1 {tid} at theta_0 (route {rr_['route']}) vs round6/results/archive_partA.jsonl + J_partA.npz (exact): {e}")
    out(f"A2-f1 the refined evaluator at f = 1 reproduces round 6 at theta_0: {'PASS' if p1 else 'FAIL'}")
    res['f1_identity'] = dict(rows=rows, pass_=p1); ok &= p1
    res['pass_'] = ok
    return res


# --- A3: small-grid explicit-matrix check on the refined grid ----------------------------------------------------------------
def dense_phys(zf, hf, xk, coarse, zc, hc, gam):
    """explicit transition matrices on the fine nodes zf with every window built from the coarse table's physical coordinates:
    node j -> the stored coarse node of smallest physical distance |zf_j - zc_i| (brute force, unique minimiser asserted),
    physical half-widths L^+- = M^+-(i) hc, omega_jl = 1 for -L^- < zf_l - zf_j < L^+, 1/2 at equality (offsets compared in
    units of hf with tolerance 1e-6), alpha = (zf_j - L^- - x_k)/sigma, beta = (zf_j + L^+ - x_k)/sigma."""
    G = zf.size; Gc = zc.size; Kn = xk.size
    dist = np.abs(zf[:, None]-zc[None, :]); ist = np.argmin(dist, axis=1)
    two = np.sort(dist, axis=1)[:, :2]; tie_margin = float(np.min(two[:, 1]-two[:, 0])/hc)
    assert tie_margin > 1e-9, "a fine node is equidistant to two coarse nodes"
    t = (zf[None, :]-zf[:, None])/hf
    g1 = np.empty(G); dr._kernel(zf, xk[0], s, hf, g1)
    m = g1.copy(); ms = [m.copy()]; EN = 1.0; F = 0.0; Ts = []; qs = []; cs = []
    for p in range(1, Kn):
        Mm, Mp = coarse.rows(p, 0, Gc-1)
        Lm = Mm[ist]*hc; Lp = Mp[ist]*hc; em = (Lm/hf)[:, None]; ep = (Lp/hf)[:, None]
        inside = (t > -em+1e-6) & (t < ep-1e-6); edge = (np.abs(t+em) <= 1e-6) | (np.abs(t-ep) <= 1e-6)
        om = inside*1.0+edge*0.5
        g = np.empty(G); dr._kernel(zf, xk[p], s, hf, g)
        Ph = om@g; qh = (1.0-om)@g
        T = np.diag(Ph)+g[None, :]*(1.0-om)
        al = (zf-Lm-xk[p])/s; be = (zf+Lp-xk[p])/s; cev, cno = dr.cfun(al, be); c = cev+cno
        EN += m@qh; F += gam[p]*(m@c)
        m = m@T; ms.append(m.copy()); Ts.append(T); qs.append(qh); cs.append(c)
    Vc = np.zeros(G); Vq = np.zeros(G); Vcs = []; Vqs = []
    for p in range(Kn-1, 0, -1):
        T = Ts[p-1]; Vc = gam[p]*cs[p-1]+T@Vc; Vq = qs[p-1]+T@Vq; Vcs.append(Vc.copy()); Vqs.append(Vq.copy())
    return dict(m_steps=ms, EN=EN, F=F, Vc=Vcs[::-1], Vq=Vqs[::-1], ENb=1.0+g1@Vq, Fb=g1@Vc, nearest=ist, tie_margin=tie_margin)

def a3(out):
    zc, hc, xk, a_, b_, tables, gam_r = ew.small_grid_setup()
    res = []; ok = True
    for f in (3, 5):
        G = ew.SG_G*f; hf = s/(24*f); zf = zc[0]+np.arange(G)*hf
        cell, gap = er.cell_index(zf, zc[0], hc)
        a = np.zeros(ew.SG_K, np.int64); b = np.full(ew.SG_K, G-1, np.int64)
        out(f"A3 f={f}: fine G {G}, h = sigma/{24*f}, z_0 = {zf[0]/s!r} sigma; min distance to a coarse cell boundary {gap!r} cell widths; "
            f"nodes beyond the last coarse cell (clipped to node {ew.SG_G-1}): {int((cell > ew.SG_G-1).sum())}")
        for tn, tab in tables.items():
            rt = er.RefinedTable(tab, f, cell)
            ft = ew.forward(zf, hf, s, xk, a, b, rt, record=True)
            for gname, gam in (('gamma=1', np.ones(ew.SG_K)), ('gamma=rand', gam_r)):
                dn = dense_phys(zf, hf, xk, tab, zc, hc, gam)
                bt = ew.backward(zf, hf, s, xk, a, b, rt, gam, record=True)
                Ff = float(gam[1:]@ft['W'][1:])
                lk = np.clip(cell, 0, ew.SG_G-1)
                d = dict(f=f, table=tn, gamma=gname, lookup_vs_dense_nearest_mismatch=int((lk != dn['nearest']).sum()), dense_tie_margin=dn['tie_margin'],
                         err_m_steps=max(relerr(x, y) for x, y in zip(ft['m_steps'], dn['m_steps'])), err_EN_fwd=relerr(ft['EN'], dn['EN']),
                         err_F_fwd=relerr(Ff, dn['F']), err_Vc_steps=max(relerr(v[2], y) for v, y in zip(bt['Vc'], dn['Vc'])),
                         err_Vq_steps=max(relerr(v[2], y) for v, y in zip(bt['Vq'], dn['Vq'])), err_EN_bwd=relerr(1.0+bt['Nq'], dn['ENb']),
                         err_F_bwd=relerr(bt['Fc'], dn['Fb']), EN=ft['EN'], F=Ff, mass_dev_max=float(np.abs(ft['mass']-1).max()),
                         min_m=float(ft['minm'].min()))
                d['pass_'] = all(d[k] <= TOL for k in d if k.startswith('err_')) and d['lookup_vs_dense_nearest_mismatch'] == 0
                ok &= d['pass_']; res.append(d)
                A4['mass_dev_max'] = max(A4['mass_dev_max'], d['mass_dev_max']); A4['min_m'] = min(A4['min_m'], d['min_m'])
                for k, v in d.items():
                    if k not in ('f', 'table', 'gamma'): out(f"A3 f={f} table={tn} {gname} {k} {v!r}")
    out(f"A3 small-grid explicit-matrix check at f = 3, 5 (3 tables x 2 gamma each, all errors <= 1e-12, lookup = dense nearest node): {'PASS' if ok else 'FAIL'}")
    return ok, res


def main(out):
    import numba, scipy
    out(f"=== checks.py (round 7 Part A); spec reports_CN/plan_round7_CN.md v3.1; python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"scipy {scipy.__version__}, numba {numba.__version__}; B = {B!r}; frozen z_0 = {er.Z0!r}, h_24 = {er.H24!r}, G_24 = {er.G24}")
    t0 = time.perf_counter(); er.warmup(); out(f"A0 warm-up (JIT) {time.perf_counter()-t0:.1f} s")
    t0 = time.perf_counter(); rc = cell_checks(out); t_c = time.perf_counter()-t0
    t0 = time.perf_counter(); r2 = a2(out); t_2 = time.perf_counter()-t0
    t0 = time.perf_counter(); ok3, r3 = a3(out); t_3 = time.perf_counter()-t0
    ok4 = A4['mass_dev_max'] <= 1e-10 and A4['min_m'] >= 0 and A4['assert_fail'] == 0 and A4['inv_all']
    out(f"A4 conservation / non-negativity / support over {A4['n']} evaluator calls + the A3 runs: max |sum m - 1| {A4['mass_dev_max']!r} "
        f"(<= 1e-10), min m {A4['min_m']!r} (>= 0), assertion failures {A4['assert_fail']}, invariants S, J, w all true {A4['inv_all']}: "
        f"{'PASS' if ok4 else 'FAIL'}")
    out(f"A-SUM cell lookup {'PASS' if rc['pass_'] else 'FAIL'}; A2 split-path regressions {'PASS' if r2['pass_'] else 'FAIL'}; "
        f"A3 explicit matrices {'PASS' if ok3 else 'FAIL'}; A4 {'PASS' if ok4 else 'FAIL'}; wall {t_c:.1f} + {t_2:.1f} + {t_3:.1f} s")
    allok = rc['pass_'] and r2['pass_'] and ok3 and ok4
    json.dump(dict(cell=rc, a2=r2, a3=r3, a3_pass=ok3, a4=dict(A4, pass_=ok4), pass_=allok, wall_s=dict(cell=t_c, a2=t_2, a3=t_3)),
              open(os.path.join(er.RES, 'partA_checks.json'), 'w'), indent=1, default=float)
    return allok


if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(er.LOGS, 'partA_checks.log'))
    t0 = time.perf_counter(); ok = main(print); print(f"=== checks.py done in {time.perf_counter()-t0:.1f} s; {'ALL PASS' if ok else 'FAILURES'}")
    sys.exit(0 if ok else 1)
