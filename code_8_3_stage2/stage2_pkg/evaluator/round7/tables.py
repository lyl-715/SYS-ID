"""Round 7, Part A0: the five frozen candidate tables with absolute coordinates; restoration of T4 (spec
reports_CN/plan_round7_CN.md v3.1, frozen; §2.1). 固定设计、离散方案下.

T4 = the deterministic policy of the round-6 m = 0.20 lambda scan at lambda/gbar = 4 (run 2 of round6/results/scan_U_W_m020.json:
nominal E[N] 262.799, Delta m_D 6.089, cost 1.741 on the sigma/24 grid). Its uint16 table was written to
round6/results/tables/U_W_m020_r02.u16.npy, which is not committed (42 MB, round 6 DECISION A-2) and is not in this clone, so
the spec's fallback applies: one DP at that exact lambda (the float of the scan record, recomputed as 0.8 * (5 gbar) exactly as
the scan formed it) with round 4's trigger_dp.py unchanged (Design('U'), gamma('W'), 400 actions, R = 88 sigma, 4 numba threads;
the argmax per node is independent of the thread count). The restored policy is regressed against the scan record (E[N], S,
F, log det, Delta m_D, dual value, mass, min m; exact) and against round 4's U/W scan run at the same lambda; the round-4
forward-backward check (B4) is repeated; the table is packed losslessly into results/table_T4.npz.

Manifest results/tables_round7.json: z_0, h_24, G_24 of the frozen nominal sigma/24 grid (dr.make_grid of the nominal output),
and per candidate the packed file, sha256, run, lambda, lambda/gbar, policy ID, support start indices (first/last) and widths.

Usage (from this directory): python3 tables.py   -> logs/partA0_tables.log, results/table_T4.npz, results/tables_round7.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import evaluator_refine as er
import trigger_dp as td                                   # round 4, unchanged
ew, dr = er.ew, er.dr
s, K, B = er.s, er.K, er.B
R4, R6 = er.R4, er.R6

SCAN = os.path.join(R6, 'results', 'scan_U_W_m020.json'); T4_RUN = 2
POLICY_ID = {'m000': 'U_W_r05 (round 4 U/W final, lambda/gbar = 3.4)', 'm005': 'U_W_m005_r11 (round 6, lambda/gbar = 3.546875)',
             'm010': 'U_W_m010_r11 (round 6, lambda/gbar = 3.715625)', 'm020': 'U_W_m020_r08 (round 6, lambda/gbar = 4.140625)',
             'T4': 'U_W_m020_r02 (round 6 m = 0.20 scan run 2, lambda/gbar = 4)'}

def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def main(out):
    import numba
    numba.set_num_threads(4)
    t0 = time.perf_counter(); td._warmup(); out(f"A0 warm-up {time.perf_counter()-t0:.1f} s; numba threads {numba.get_num_threads()}")
    des = td.Design('U', out); gam, gbar, g1 = des.gamma('W')
    st = json.load(open(SCAN)); r2 = [r for r in st['runs'] if r['run'] == T4_RUN][0]
    lam = 0.8*(5.0*gbar)
    out(f"A0 gbar {gbar!r} (scan record {st['gbar']!r}, equal {gbar == st['gbar']}); lambda = 0.8 * (5 gbar) = {lam!r} "
        f"(scan run {T4_RUN} {r2['lam']!r}, equal {lam == r2['lam']}); lambda/gbar {lam/gbar!r}")
    assert gbar == st['gbar'] and lam == r2['lam'] and st['B'] == (1.0-0.20)*B
    exist = os.path.exists(r2['table'])
    out(f"A0 scan checkpoint table {r2['table']} present in this clone: {exist} (not committed; round 6 DECISION A-2) -> "
        f"{'restore from it' if exist else 'one DP at the exact lambda (spec §2.1 fallback)'}")
    t0 = time.perf_counter()
    if exist: codes = np.array(np.load(r2['table'])); dsum = None; how = 'restored from the scan checkpoint'
    else:
        codes = np.empty((K-1, des.WMAX), np.uint16); dsum, nbad = des.dp(gam, lam, codes); how = 'DP re-run at the exact lambda'
        out(f"A0 DP at lambda {lam!r}: {time.perf_counter()-t0:.1f} s, non-finite V {nbad}")
    tab = des.table(codes)
    rec, J, rr = ew.evaluate_table('U', tab)
    _, J_sod, _ = des.sod_reference(); ld_sod = float(np.linalg.slogdet(J_sod)[1])
    F = float(gam[1:]@rr['W'][1:]); EN = float(rec['EN']); N = EN-1.0
    reg = dict(EN=EN == r2['EN'], S=float(rec['S']) == r2['S'], F=F == r2['F'], logdet=float(rec['logdet']) == r2['logdet'],
               dmD=float(rec['logdet'])-ld_sod == r2['dmD'], mass_dev_max=float(rec['mass_dev_max']) == r2['mass_dev_max'],
               min_m=float(rec['min_m']) == r2['min_m'])
    if dsum is not None:
        dual = g1+dsum+lam*(st['B']-1.0); reg['dual'] = dual == r2['dual']
        reg['dp_minus_policy'] = (dsum-(F-lam*N)) == r2['dp_minus_policy']
    for k, v in reg.items(): out(f"A0-REG T4 {k} equal to scan_U_W_m020.json run {T4_RUN}: {v}")
    out(f"A0-REG T4 E[N] {EN!r} S {float(rec['S'])!r} F {F!r} log det {float(rec['logdet'])!r} Delta m_D {float(rec['logdet'])-ld_sod!r}")
    st4 = json.load(open(os.path.join(R4, 'results', 'scan_U_W.json')))
    r4 = [r for r in st4['runs'] if r['lam'] == lam]
    if r4: out(f"A0-REG T4 vs round 4 scan_U_W run {r4[0]['run']} at the same lambda: E[N] equal {EN == r4[0]['EN']}, log det equal {float(rec['logdet']) == r4[0]['logdet']}")
    else: out("A0-REG round 4 scan_U_W has no run at this lambda")
    fw = ew.forward(des.z, des.dz, s, des.xk, des.a, des.b, tab, gam=gam)
    bw = ew.backward(des.z, des.dz, s, des.xk, des.a, des.b, tab, gam)
    eF = abs(bw['Fc']-fw['Fgam'])/abs(fw['Fgam']); eN = abs(bw['Nq']-(fw['EN']-1.0))/abs(fw['EN']-1.0)
    out(f"A0-B4 T4 forward-backward: F rel {eF!r}, N rel {eN!r} (<= 1e-10: {eF <= 1e-10 and eN <= 1e-10}); forward E[N] equal {fw['EN'] == EN}")
    ok = all(reg.values()) and eF <= 1e-10 and eN <= 1e-10
    pk = er.TABLE_NPZ['T4']
    np.savez_compressed(pk, codes=codes, start=des.a[1:], width=des.w[1:], lam=lam, run=T4_RUN, lam_over_gbar=lam/gbar,
                        note=np.array('round 7 T4 = round 6 m = 0.20 scan run 2 (lambda = 0.8 * 5 gbar), ' + how + '; layout as '
                                      'round4/results/table_<case>.npz: codes[r, o] = action of step k = r + 2 at node start[r] + o '
                                      '(o < width[r]); code = i_- * 20 + i_+, (M^-, M^+) = (48 (i_- + 1), 48 (i_+ + 1)) sigma/24 units; pad 65535'))
    with np.load(pk) as f_: rt = bool(np.array_equal(f_['codes'], codes))
    out(f"A0 T4 packed {os.path.relpath(pk, HERE)} ({os.path.getsize(pk)} bytes), round trip equal {rt}")
    man = dict(grid=dict(z0=er.Z0, h24=er.H24, G24=er.G24, definition='dr.make_grid(xbar_nominal, 24, 8 Delta, B): z_i = z_0 + i h_24, '
                         'z_0 = min xbar - 8 Delta; quadrature grid of factor f: dr.make_grid(xbar_nominal, 24 f, 8 Delta, B)'),
               B=B, tables={})
    a, b = ew.windows88(er.I24['xbar'], er.Z0, er.H24, er.G24)
    for tid, p in er.TABLE_NPZ.items():
        with np.load(p) as f_: start, width, lam_, run_ = f_['start'], f_['width'], float(f_['lam']), int(f_['run'])
        assert np.array_equal(start, a[1:]) and np.array_equal(width, (b-a+1)[1:])
        man['tables'][tid] = dict(policy_id=POLICY_ID[tid], file=os.path.relpath(p, HERE), sha256=sha(p), run=run_, lam=lam_, lam_over_gbar=lam_/gbar,
                                  start_first=int(start[0]), start_last=int(start[-1]), width_min=int(width.min()), width_max=int(width.max()),
                                  steps=int(len(start)))
        out(f"A0-MAN {tid} {man['tables'][tid]}")
    man['T4_restore'] = dict(how=how, regression=reg, fb=dict(F_rel=eF, N_rel=eN), EN=EN, S=float(rec['S']), F=F, logdet=float(rec['logdet']),
                             dmD=float(rec['logdet'])-ld_sod, cost_vs_m000_sigma24=7.830436416895992-(float(rec['logdet'])-ld_sod), pass_=bool(ok and rt))
    json.dump(man, open(os.path.join(er.RES, 'tables_round7.json'), 'w'), indent=1)
    out(f"A0 T4 restoration ({how}) and regression: {'PASS' if ok and rt else 'FAIL'}")
    return ok and rt


if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(er.LOGS, 'partA0_tables.log'))
    print(f"=== tables.py started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round7_CN.md v3.1 §2.1; B = {B!r}")
    t0 = time.perf_counter(); ok = main(print); print(f"=== tables.py done in {time.perf_counter()-t0:.1f} s")
    sys.exit(0 if ok else 1)
