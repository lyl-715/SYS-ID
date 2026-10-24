"""Round 6, Part A1: budget-margin designs of the uniform-input weighted table (spec reports_CN/plan_round6_CN.md v3.2, frozen;
§1.1, §4 items 2-3). 固定设计、离散方案下 -- discrete-MDP designs evaluated by the discrete evaluator, not bounds.

round4/trigger_dp.py is reused unchanged (Design, gamma('W') with A = J_SoD(u_unif)^-1, Scan with the round-5 Part 0a doubling
rule, final policy = largest F among the evaluated feasible policies, run cap 40). Only the feasibility target changes: the
module-level budget is set to B_m = (1 - m) B for the scan of margin m (as round5/two_stage.py scheme D set it to 2B), with
the scan record, tables and runs pointed at round6/results/ (DECISION A-1). The DP itself (action set, grids, first/last-step
rule, reward) does not see B; the dual value logged per lambda uses B_m (lambda (B_m - 1)).

  m in {0.05, 0.10, 0.20}  ->  results/scan_U_W_m{005,010,020}.json, logs/scan_U_W_m{005,010,020}.log (one LAM line per DP run)
  final tables             ->  results/table_U_W_m{005,010,020}.npz (codes, support start indices, widths, lambda, run; lossless)
  m = 0                    ->  the round-4 U/weighted table, round4/results/table_U_W.npz (not re-designed)

Usage (from this directory):
  python3 margin_design.py scan [--margin m005 ...]   # lambda scans; checkpoint after every DP run, a re-run resumes
  python3 margin_design.py final                      # per-table report rows, forward-backward check, packed npz
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); R4 = os.path.join(os.path.dirname(HERE), 'round4')
sys.path.insert(0, R4)
import numpy as np
import evaluator_window as ew
import trigger_dp as td
dr = ew.dr

s, K = ew.s, ew.K
B = ew.B_BUDGET
assert B == 316.8921168493721
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
TABLES = os.path.join(RES, 'tables'); RUNS = os.path.join(RES, 'runs')
for _d in (LOGS, RES, TABLES, RUNS): os.makedirs(_d, exist_ok=True)
MARGINS = {'m005': 0.05, 'm010': 0.10, 'm020': 0.20}
VERDICT = '离散方案下'


def B_m(m): return (1.0-m)*B

def tag_of(mt): return f'U_W_{mt}'

def paths(mt):
    tag = tag_of(mt)
    return tag, os.path.join(RES, f'scan_{tag}.json'), os.path.join(LOGS, f'scan_{tag}.log')

def point_td(mt):
    """trigger_dp's module-level budget and output paths for the scan of margin mt (restored by the caller's process end)."""
    td.B = B_m(MARGINS[mt]); td.TABLES = TABLES; td.RUNS = RUNS
    td.case_paths = lambda case, _mt=mt: paths(_mt)


def scan(margins, log_main):
    t0 = time.perf_counter(); td._warmup(); log_main(f"WARMUP {time.perf_counter()-t0:.1f} s")
    des = td.Design('U', log_main)
    g4 = json.load(open(os.path.join(R4, 'results', 'scan_U_W.json')))['gbar']
    out = {}
    for mt in margins:
        tag, jpath, lpath = paths(mt)
        point_td(mt)
        f = open(lpath, 'a')
        def log(msg, _f=f): print(msg); _f.write(msg+'\n'); _f.flush()
        log(f"=== margin_design.py scan {mt} started {time.strftime('%Y-%m-%d %H:%M:%S')}; m = {MARGINS[mt]!r}; B = {B!r}; "
            f"B_m = (1 - m) B = {td.B!r}; trigger_dp unchanged (actions 400, M^+- = 48..960, R = 88 sigma, lambda rule of round 4)")
        t1 = time.perf_counter()
        sc = td.Scan(des, 'W', log)
        log(f"GAMMA {mt} gbar {sc.gbar!r} (round 4 U/W {g4!r}, equal {sc.gbar == g4})")
        st = sc.scan()
        wall = time.perf_counter()-t1
        log(f"END {mt} status {st['status']}; DP runs {len(st['runs'])}; wall {wall:.1f} s")
        st['wall_s'] = wall; st['B_m'] = td.B; st['m'] = MARGINS[mt]
        td._atomic_json(st, jpath)
        out[mt] = st; f.close()
    return out


def final(log):
    """per margin: final run of the scan, forward-backward consistency (round-4 B4 check), packed npz with round trip;
    the m = 0 row from the round-4 records."""
    td._warmup()
    des = td.Design('U', log); gam, gbar, g1 = des.gamma('W')
    rows = {}
    st4 = json.load(open(os.path.join(R4, 'results', 'scan_U_W.json'))); fc4 = json.load(open(os.path.join(R4, 'results', 'final_checks.json')))['U/W']
    r4 = [x for x in st4['runs'] if x['run'] == st4['final_run']][0]
    rows['m000'] = dict(m=0.0, B_m=B, run=r4['run'], runs=len(st4['runs']), lam=r4['lam'], lam_over_gbar=r4['lam_over_gbar'], EN=r4['EN'],
                        S=r4['S'], F=r4['F'], logdet=r4['logdet'], dmD=r4['dmD'], slack_Bm=B-r4['EN'], slack_B=B-r4['EN'],
                        feasible_Bm=bool(r4['EN'] <= B), utilisation_target_met=bool(B-r4['EN'] <= 0.5), status=st4['status'],
                        notes=st4.get('notes'), fb_pass=fc4['fb_pass'], F_rel=fc4['F_rel'], N_rel=fc4['N_rel'],
                        mass_dev_max=r4['mass_dev_max'], min_m=r4['min_m'], assert_fail=r4['assert_fail'], mono_worst=st4.get('mono_worst'),
                        packed=os.path.relpath(os.path.join(R4, 'results', 'table_U_W.npz'), HERE), source='round 4 (not re-designed)')
    for mt, m in MARGINS.items():
        tag, jpath, _ = paths(mt)
        st = json.load(open(jpath)); fr = st.get('final_run'); Bm = B_m(m)
        assert st['B'] == Bm, (st['B'], Bm)
        if fr is None:
            rows[mt] = dict(m=m, B_m=Bm, status=st['status'], message=st.get('message'), runs=len(st['runs']))
            log(f"A1-FINAL {mt} no final policy (status {st['status']})"); continue
        r = [x for x in st['runs'] if x['run'] == fr][0]
        codes = np.load(r['table'], mmap_mode='r'); tab = des.table(codes)
        t0 = time.perf_counter()
        fw = ew.forward(des.z, des.dz, s, des.xk, des.a, des.b, tab, gam=gam)
        bw = ew.backward(des.z, des.dz, s, des.xk, des.a, des.b, tab, gam)
        t_c = time.perf_counter()-t0
        Ff = fw['Fgam']; Nf = fw['EN']-1.0
        eF = abs(bw['Fc']-Ff)/abs(Ff); eN = abs(bw['Nq']-Nf)/abs(Nf)
        packed = os.path.join(RES, f'table_{tag}.npz')
        np.savez_compressed(packed, codes=np.asarray(codes), start=des.a[1:], width=des.w[1:], lam=r['lam'], run=fr, m=m, B_m=Bm,
                            note=np.array(f'round 6 margin table m = {m}: U/W objective, feasibility target B_m = (1 - m) B; layout as '
                                          'round4/results/table_<case>.npz: codes[r, o] = action of step k = r + 2 at node start[r] + o '
                                          '(o < width[r]); code = i_- * 20 + i_+, (M^-, M^+) = (48 (i_- + 1), 48 (i_+ + 1)); pad 65535'))
        with np.load(packed) as f_: rt = bool(np.array_equal(f_['codes'], np.asarray(codes)))
        rows[mt] = dict(m=m, B_m=Bm, run=fr, runs=len(st['runs']), lam=r['lam'], lam_over_gbar=r['lam_over_gbar'], EN=r['EN'], S=r['S'],
                        F=r['F'], logdet=r['logdet'], dmD=r['dmD'], slack_Bm=Bm-r['EN'], slack_B=B-r['EN'], feasible_Bm=bool(r['EN'] <= Bm),
                        utilisation_target_met=bool(Bm-r['EN'] <= 0.5), status=st['status'], notes=st.get('notes'),
                        fb_pass=bool(eF <= 1e-10 and eN <= 1e-10), F_rel=eF, N_rel=eN, EN_forward_same_as_scan=bool(fw['EN'] == r['EN']),
                        mass_dev_max=r['mass_dev_max'], min_m=r['min_m'], assert_fail=r['assert_fail'], mono_worst=st.get('mono_worst'),
                        packed=os.path.relpath(packed, HERE), packed_bytes=os.path.getsize(packed), packed_roundtrip_equal=rt,
                        check_s=t_c, scan_wall_s=st.get('wall_s'), lam_bracket=st.get('lam_bracket'), min_dual=st.get('min_dual'))
    for mt, d in rows.items():
        for k, v in d.items(): log(f"A1 {mt} {k} {v!r}")
    log(f"A1-TAB {'table':5s} {'m':>5s} {'B_m':>12s} {'run':>3s} {'runs':>4s} {'lam/gbar':>10s} {'E[N](theta0)':>14s} {'B_m-E[N]':>10s} "
        f"{'<=0.5':>5s} {'B-E[N]':>9s} {'S':>9s} {'logdet':>10s} {'dmD':>8s} {'fb':>4s}")
    for mt, d in rows.items():
        if 'EN' not in d: log(f"A1-TAB {mt:5s} no final policy"); continue
        log(f"A1-TAB {mt:5s} {d['m']:5.2f} {d['B_m']:12.6f} {d['run']:3d} {d['runs']:4d} {d['lam_over_gbar']:10.6f} {d['EN']:14.6f} "
            f"{d['slack_Bm']:10.6f} {('yes' if d['utilisation_target_met'] else 'no'):>5s} {d['slack_B']:9.4f} {d['S']:9.3f} {d['logdet']:10.5f} "
            f"{d['dmD']:+8.5f} {('ok' if d['fb_pass'] else 'FAIL'):>4s}")
    td._atomic_json(rows, os.path.join(RES, 'margin_final.json'))
    return rows


if __name__ == '__main__':
    import numba
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['scan', 'final']); ap.add_argument('--margin', action='append')
    ap.add_argument('--threads', type=int, default=4)
    A_ = ap.parse_args()
    numba.set_num_threads(A_.threads)
    sys.stdout = dr.Tee(os.path.join(LOGS, f'partA1_{A_.stage}.log'), 'a' if A_.stage == 'scan' else 'w')
    import scipy
    print(f"=== margin_design.py {A_.stage} started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round6_CN.md v3.2; python "
          f"{sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, numba {numba.__version__} ({numba.get_num_threads()} threads); "
          f"B = {B!r}")
    t0 = time.perf_counter()
    if A_.stage == 'scan': scan(A_.margin or list(MARGINS), print)
    else: final(print)
    print(f"=== margin_design.py {A_.stage} done in {time.perf_counter()-t0:.1f} s")
