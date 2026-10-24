"""Round 5, Part B: fixed-design two-stage comparison (spec reports_CN/plan_round5_CN.md v3.2, frozen; §1.3, §2 item 4).
固定设计的两段式比较 -- two tables fixed in advance, two independent 5-second experiments at the same true theta, information
matrices added. Not the sequential adaptive scheme (estimate after stage 1, then build the table), which is not evaluated here.

  A   one 5-s stage, constant Delta_A = M h (uniform input, sigma = Delta*/20 fixed, h = sigma/24): every integer M = 240..480
      evaluated (241 forward passes; no monotonicity of E[N] in M assumed), M_A = argmax{E[N_M] : E[N_M] <= 2B}, ties -> larger M
  A'  SoD + SoD: 2 J_SoD(Delta*); sends 2 E[N_SoD(theta)] (not 2B by assumption)
  B   SoD + the round-4 U/weighted table (budget B): J_SoD + J_UW; sends E[N_SoD(theta)] + E[N_UW(theta)]
  D   the U/weighted design (same objective, same fixed gamma_k = sigma^-2 h_k^T J_SoD^-1 h_k) with budget 2B, lambda re-scanned
      by the round-4 rule (trigger_dp.Scan with the Part 0a doubling); the only re-design of this round, every lambda logged
Budget: the SUM E[N_1(theta)] + E[N_2(theta)] <= 2B, 2B = 633.7842336987442, recorded with each stage's sends and the slack;
"each stage <= B" is a separate constraint, logged separately. Verdicts are those of the discrete evaluator (离散方案下).
Evaluated at theta_0 (A, A', B, D) and at the P4 points with eps in {0.2, 0.5} (A' and B only, U's perturbed parameters, from
the Part A archive: the same SoD and U/W evaluations, J's from results/J_partA.npz).

Usage (from this directory):
  python3 two_stage.py schemeA            # 241 forwards, 4 workers -> results/schemeA.json, logs/partB_schemeA.log
  python3 two_stage.py schemeD-redesign   # lambda scan with budget 2B from an EMPTY checkpoint (numba, 4 threads) ->
                                          # results/scan_U_W_2B.json, results/table_U_W_2B.npz, results/schemeD.json
  python3 two_stage.py schemeD-load       # load the committed results/table_U_W_2B.npz, recompute the final numbers by one
                                          # forward evaluation, compare with results/schemeD.json -> results/schemeD_load.json
  python3 two_stage.py combine            # theta_0 table and P4 table -> results/partB.json, logs/partB_combine.log
Round 6 Part 0b: the former single stage 'schemeD' resumed from whatever checkpoint was present, including uint16 tables
under results/tables/ that are not committed; it is split into the two explicit entry points above. schemeD-redesign refuses
to start when a scan record or a table of this case exists; schemeD-load reads only committed files (checked with git).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); R4 = os.path.join(os.path.dirname(HERE), 'round4')
sys.path.insert(0, R4)
import numpy as np
import evaluator_window as ew
dr = ew.dr

s, K, B = ew.s, ew.K, ew.B_BUDGET
B2 = 2*B
assert B2 == 633.7842336987442
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)
M_RANGE = range(240, 481)
VERDICT = '离散方案下'

def logdet(J): sg, ld = np.linalg.slogdet(J); assert sg > 0; return float(ld)


# --- scheme A: the E[N_M] curve -----------------------------------------------------------------------------------------
def _init(): ew._warmup()

def _evalM(M):
    t0 = time.perf_counter()
    rec, J, r = ew.evaluate_table('U', ew.ConstTable(M, M))
    return dict(M=M, Delta_over_sigma=M/ew.N_GRID, EN=float(rec['EN']), EN_P=float(rec['EN_P']), S=float(rec['S']), logdet=float(rec['logdet']),
                mass_dev_max=float(rec['mass_dev_max']), min_m=float(rec['min_m']), assert_fail=int((~r['assert_ok']).sum()),
                path=rec['path'], wall_s=time.perf_counter()-t0), J

def schemeA(out, workers=4):
    import multiprocessing as mp
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(workers, initializer=_init) as pool:
        res = pool.map(_evalM, list(M_RANGE), chunksize=1)
    wall = time.perf_counter()-t0
    rows = [r for r, J in res]; Js = {r['M']: J for r, J in res}
    out(f"A-CURVE {'M':>4s} {'Delta/sigma':>11s} {'E[N_M]':>18s} {'E[N_M]<=2B':>10s} {'S':>10s} {'logdet':>11s} path")
    for r in rows:
        out(f"A-CURVE {r['M']:4d} {r['Delta_over_sigma']:11.6f} {r['EN']!r:>18s} {('yes' if r['EN'] <= B2 else 'NO'):>10s} {r['S']:10.3f} "
            f"{r['logdet']:11.5f} {r['path']}")
    EN = np.array([r['EN'] for r in rows]); Ms = np.array([r['M'] for r in rows])
    inc = int((np.diff(EN) > 0).sum())
    out(f"A-CURVE non-monotone steps (E[N_(M+1)] > E[N_M]) over M = 240..480: {inc}; E[N] at M = 240 {EN[0]!r}, at M = 480 {EN[-1]!r}")
    feas = [r for r in rows if r['EN'] <= B2]
    if feas:
        best = max(r['EN'] for r in feas); MA = max(r['M'] for r in feas if r['EN'] == best)
        rA = [r for r in rows if r['M'] == MA][0]; slack = B2-rA['EN']
        out(f"A-SEL M_A = {MA} (Delta_A = {MA/ew.N_GRID!r} sigma), E[N_M_A] = {rA['EN']!r}, slack to 2B = {slack!r}; "
            f"{'slack > 5 sends: reported as is, range not widened' if slack > 5 else 'slack <= 5 sends'}")
        np.save(os.path.join(RES, 'J_schemeA.npy'), Js[MA])
    else:
        MA, slack = None, None; out("A-SEL no M in 240..480 with E[N_M] <= 2B (range not widened)")
    out(f"A-SUM 241 forwards in {wall:.1f} s with {workers} workers; max |sum m - 1| {max(r['mass_dev_max'] for r in rows)!r}; "
        f"min m {min(r['min_m'] for r in rows)!r}; assertion failures {sum(r['assert_fail'] for r in rows)}")
    json.dump(dict(rows=rows, M_A=MA, slack=slack, nonmonotone_steps=inc, wall_s=wall, workers=workers, B2=B2),
              open(os.path.join(RES, 'schemeA.json'), 'w'), indent=1)


# --- scheme D: lambda re-scan with budget 2B ------------------------------------------------------------------------------
TAG_D = 'U_W_2B'
SCAN_D = os.path.join(RES, f'scan_{TAG_D}.json'); PACKED_D = os.path.join(RES, f'table_{TAG_D}.npz')
JSON_D = os.path.join(RES, 'schemeD.json'); J_D = os.path.join(RES, 'J_schemeD.npy')

def schemeD_redesign(out, threads=4):
    """entry point 1: the lambda scan from an empty checkpoint. Stops before any DP if a scan record or a run/table file of
    this case exists (move them aside to redesign); never resumes."""
    import glob
    present = [p for p in [SCAN_D]+glob.glob(os.path.join(RES, 'tables', f'{TAG_D}_r*'))+glob.glob(os.path.join(RES, 'runs', f'{TAG_D}_r*'))
               if os.path.exists(p)]
    if present:
        out(f"D-REDESIGN checkpoint not empty, stopping before any DP (move these aside to redesign): {[os.path.relpath(p, HERE) for p in present]}")
        return None
    return _schemeD_scan(out, threads)

def _schemeD_scan(out, threads):
    import numba
    import trigger_dp as td
    numba.set_num_threads(threads)
    tag = TAG_D; jpath = SCAN_D
    # trigger_dp is reused unchanged; its module-level budget and output paths are pointed at round 5 for this run only
    # (DECISION B-3): B -> 2B, tables/runs -> round5/results/{tables,runs}, scan record -> round5/results/scan_U_W_2B.json
    td.B = B2
    td.TABLES = os.path.join(RES, 'tables'); td.RUNS = os.path.join(RES, 'runs')
    os.makedirs(td.TABLES, exist_ok=True); os.makedirs(td.RUNS, exist_ok=True)
    td.case_paths = lambda case: (tag, jpath, os.path.join(LOGS, f'scan_{tag}.log'))
    out(f"D-SETUP trigger_dp.Scan on U/W with budget 2B = {td.B!r}; numba threads {numba.get_num_threads()}; lambda rule of round 4 "
        f"(lambda_0 = 5 gbar, doubling to the cap {td.LAM_CAP} gbar, x0.8 down to the first infeasible, 8 bisections, final = largest F)")
    t0 = time.perf_counter(); td._warmup(); out(f"D-WARMUP {time.perf_counter()-t0:.1f} s")
    t0 = time.perf_counter()
    des = td.Design('U', out)
    gam, gbar, g1 = des.gamma('W')
    g4 = json.load(open(os.path.join(R4, 'results', 'scan_U_W.json')))['gbar']
    out(f"D-GAMMA gbar {gbar!r} (round 4 U/W {g4!r}, equal {gbar == g4}); gamma_1 {g1!r}")
    st = td.Scan(des, 'W', out).scan()
    wall = time.perf_counter()-t0
    out(f"D-END status {st['status']}; DP runs {len(st['runs'])}; wall {wall:.1f} s (design cache, scan; warm-up excluded)")
    fr = st.get('final_run')
    res = dict(status=st['status'], runs=len(st['runs']), final_run=fr, wall_s=wall, gbar=gbar, gbar_round4=g4)
    if fr is not None:
        r = [x for x in st['runs'] if x['run'] == fr][0]
        J = np.load(os.path.join(td.RUNS, f"{tag}_r{fr:02d}.npz"))['J']; np.save(os.path.join(RES, 'J_schemeD.npy'), J)
        codes = np.load(r['table'], mmap_mode='r')
        packed = os.path.join(RES, f'table_{tag}.npz')
        np.savez_compressed(packed, codes=np.asarray(codes), start=des.a[1:], width=des.w[1:], lam=r['lam'], run=fr,
                            note=np.array('scheme D (round 5): U/W objective, budget 2B; layout as round4/results/table_<case>.npz'))
        with np.load(packed) as f: rt = bool(np.array_equal(f['codes'], np.asarray(codes)))
        res.update(lam=r['lam'], lam_over_gbar=r['lam_over_gbar'], EN=r['EN'], S=r['S'], F=r['F'], logdet=logdet(J), slack=B2-r['EN'],
                   feasible=r['feasible'], packed=os.path.relpath(packed, HERE), packed_roundtrip_equal=rt,
                   mass_dev_max=r['mass_dev_max'], min_m=r['min_m'], assert_fail=r['assert_fail'], mono_worst=st.get('mono_worst'))
        for k, v in res.items(): out(f"D-FINAL {k} {v!r}")
    json.dump(res, open(os.path.join(RES, 'schemeD.json'), 'w'), indent=1)


def _git_tracked(path):
    """True / False if git answers, None without git (logged, not asserted then)."""
    import subprocess
    try:
        r = subprocess.run(['git', 'ls-files', '--error-unmatch', os.path.basename(path)], cwd=os.path.dirname(path),
                           capture_output=True, text=True)
        return r.returncode == 0
    except OSError:
        return None

def schemeD_load(out):
    """entry point 2: load the committed results/table_U_W_2B.npz (and the committed scan record for its run number and lambda),
    recompute the final numbers by one forward evaluation (round-4 evaluator_window.evaluate_table on a CodeTable built from
    the npz; gamma from trigger_dp.Design, which evaluates the SoD in memory), compare with results/schemeD.json and
    results/J_schemeD.npy. Reads no path under results/tables/ or results/runs/."""
    import trigger_dp as td
    reads = (SCAN_D, PACKED_D, JSON_D, J_D)
    for p in reads:
        tr = _git_tracked(p); out(f"D-LOAD reads {os.path.relpath(p, HERE)}; tracked by git: {tr}")
        assert tr is not False, f"{p} is not a committed file"
    st = json.load(open(SCAN_D)); fr = st['final_run']; r = [x for x in st['runs'] if x['run'] == fr][0]
    ref = json.load(open(JSON_D)); Jref = np.load(J_D)
    with np.load(PACKED_D) as f:
        codes, start, width, lam, run = f['codes'], f['start'], f['width'], float(f['lam']), int(f['run'])
    assert run == fr and lam == r['lam'] and run == ref['final_run'] and lam == ref['lam'], (run, fr, lam, r['lam'])
    t0 = time.perf_counter(); ew._warmup()
    des = td.Design('U', out)
    assert np.array_equal(start, des.a[1:]) and np.array_equal(width, des.w[1:]), "npz support ranges differ from the design grid"
    gam, gbar, g1 = des.gamma('W')
    tab = ew.CodeTable(codes, start, width)
    rec, J, rr = ew.evaluate_table('U', tab)
    W = rr['W']; F = float(gam[1:]@W[1:]); ld = logdet(J)
    res = dict(run=run, lam=lam, EN=float(rec['EN']), S=float(rec['S']), F=F, logdet=ld, logdet_record=float(rec['logdet']),
               slack=B2-float(rec['EN']), feasible=bool(rec['EN'] <= B2), mass_dev_max=float(rec['mass_dev_max']), min_m=float(rec['min_m']),
               assert_fail=int((~rr['assert_ok']).sum()), path=rec['path'], gbar=gbar, wall_s=time.perf_counter()-t0)
    cmp = {k: (res[k], ref[k], res[k] == ref[k]) for k in ('EN', 'S', 'F', 'logdet', 'slack', 'feasible', 'gbar')}
    cmp['logdet_record_vs_scan'] = (res['logdet_record'], r['logdet'], res['logdet_record'] == r['logdet'])
    cmp['EN_vs_scan'] = (res['EN'], r['EN'], res['EN'] == r['EN'])
    jeq = bool(np.array_equal(J, Jref)); res['J_bitwise_vs_J_schemeD'] = jeq
    for k, (a, b, e) in cmp.items(): out(f"D-LOAD {k} recomputed {a!r} committed {b!r} bitwise equal {e}")
    out(f"D-LOAD J (81 elements) bitwise equal to results/J_schemeD.npy: {jeq}")
    ok = all(e for a, b, e in cmp.values()) and jeq
    res['bitwise_all'] = ok; res['compare'] = {k: dict(recomputed=a, committed=b, equal=e) for k, (a, b, e) in cmp.items()}
    out(f"D-LOAD scheme D final numbers reproduced from the committed npz bit for bit (E[N], S, F, log det, 81 J): {'PASS' if ok else 'FAIL'}; "
        f"{res['wall_s']:.1f} s")
    json.dump(res, open(os.path.join(RES, 'schemeD_load.json'), 'w'), indent=1, ensure_ascii=False)
    return ok


# --- combine: theta_0 table and P4 table ---------------------------------------------------------------------------------------
def combine(out):
    A = json.load(open(os.path.join(RES, 'schemeA.json'))); Dd = json.load(open(os.path.join(RES, 'schemeD.json')))
    c1 = json.load(open(os.path.join(R4, 'results', 'c1.json')))
    Jsod = np.load(os.path.join(R4, 'results', 'J_U_SoD.npy')); Juw = np.load(os.path.join(R4, 'results', 'J_U_W.npy'))
    ENsod, ENuw = c1['U/SoD']['EN'], c1['U/W']['EN']
    rA = [r for r in A['rows'] if r['M'] == A['M_A']][0]; JA = np.load(os.path.join(RES, 'J_schemeA.npy'))
    ldAp = logdet(2*Jsod); ldB = logdet(Jsod+Juw); ldA = logdet(JA)
    assert abs(rA['logdet']-ldA) == 0.0
    rows = {
        'A':  dict(stage1=rA['EN'], stage2=None, total=rA['EN'], logdet=ldA, note=f"M_A = {A['M_A']}, Delta_A = {A['M_A']/ew.N_GRID} sigma"),
        "A'": dict(stage1=ENsod, stage2=ENsod, total=2*ENsod, logdet=ldAp, note='SoD + SoD'),
        'B':  dict(stage1=ENsod, stage2=ENuw, total=ENsod+ENuw, logdet=ldB, note='SoD + round-4 U/W table'),
        'D':  dict(stage1=Dd.get('EN'), stage2=None, total=Dd.get('EN'), logdet=Dd.get('logdet'), note='U/W design, budget 2B'),
    }
    out(f"B0 {'scheme':6s} {'stage 1':>12s} {'stage 2':>12s} {'total':>12s} {'slack to 2B':>12s} {'sum<=2B':>8s} {'each<=B':>8s} "
        f"{'logdet':>11s} {'vs A-prime':>11s}  note")
    for k, r in rows.items():
        if r['total'] is None: out(f"B0 {k:6s} not available ({r['note']})"); continue
        r['slack'] = B2-r['total']; r['sum_ok'] = bool(r['total'] <= B2); r['dmD_vs_Ap'] = r['logdet']-ldAp
        r['each_ok'] = (bool(r['stage1'] <= B and r['stage2'] <= B) if r['stage2'] is not None else None)
        out(f"B0 {k:6s} {r['stage1']:12.4f} {(r['stage2'] if r['stage2'] is not None else float('nan')):12.4f} {r['total']:12.4f} "
            f"{r['slack']:12.4f} {('yes' if r['sum_ok'] else 'NO'):>8s} {('n/a' if r['each_ok'] is None else ('yes' if r['each_ok'] else 'NO')):>8s} "
            f"{r['logdet']:11.5f} {r['dmD_vs_Ap']:+11.5f}  {r['note']}")
        for kk in ('stage1', 'stage2', 'total', 'slack', 'logdet', 'dmD_vs_Ap'):
            out(f"B0 {k} {kk} {r[kk]!r}")
    out(f"B0 check (known baselines, not predictions): log det A' = {ldAp!r} (203.439: {round(ldAp, 3) == 203.439}); "
        f"log det B = {ldB!r} (208.292: {round(ldB, 3) == 208.292}); B - A' = {ldB-ldAp!r} (4.852: {round(ldB-ldAp, 3) == 4.852})")
    # P4 points, eps in {0.2, 0.5}: A' and B from the Part A archive
    arch = [json.loads(l) for l in open(os.path.join(RES, 'archive_partA.jsonl'))]
    Jz = np.load(os.path.join(RES, 'J_partA.npz'))
    p4 = []
    out(f"B4 {'eps':>4s} {'dir':4s} {'N_SoD':>10s} {'N_UW':>10s} {'B total':>10s} {'slack':>9s} {'sum<=2B':>7s} {'each<=B':>7s} "
        f"{'A-prime total':>13s} {'sum<=2B':>7s} {'logdet A-prime':>14s} {'logdet B':>10s} {'B vs A-prime':>12s}")
    for e in (0.2, 0.5):
        for j in range(1, 11):
            sel = {r['table']: r for r in arch if r['input'] == 'U' and r['cls'] == 'P4' and r['eps_target'] == e and r['dir'] == f'd{j}'}
            rs, ru = sel['SoD'], sel['W']
            if rs['status'] != 'ok' or ru['status'] != 'ok':
                out(f"B4 {e} d{j} not available (assertion stop in Part A)"); continue
            Js, Ju = Jz[rs['id']], Jz[ru['id']]
            lAp, lB = logdet(2*Js), logdet(Js+Ju)
            d = dict(eps=e, dir=f'd{j}', theta=ru['theta'], eps_attained=ru['eps_attained'], N_SoD=rs['EN'], N_UW=ru['EN'],
                     B_total=rs['EN']+ru['EN'], B_slack=B2-(rs['EN']+ru['EN']), B_sum_ok=bool(rs['EN']+ru['EN'] <= B2),
                     B_each_ok=bool(rs['EN'] <= B and ru['EN'] <= B), Ap_total=2*rs['EN'], Ap_slack=B2-2*rs['EN'],
                     Ap_sum_ok=bool(2*rs['EN'] <= B2), logdet_Ap=lAp, logdet_B=lB, B_vs_Ap=lB-lAp, verdict_label=VERDICT)
            p4.append(d)
            out(f"B4 {e:4.1f} d{j:<3d} {d['N_SoD']:10.4f} {d['N_UW']:10.4f} {d['B_total']:10.4f} {d['B_slack']:+9.4f} "
                f"{('yes' if d['B_sum_ok'] else 'NO'):>7s} {('yes' if d['B_each_ok'] else 'NO'):>7s} {d['Ap_total']:13.4f} "
                f"{('yes' if d['Ap_sum_ok'] else 'NO'):>7s} {lAp:14.5f} {lB:10.5f} {lB-lAp:+12.5f}")
    for e in (0.2, 0.5):
        q = [d for d in p4 if d['eps'] == e]
        if q:
            out(f"B4-SUM eps={e}: median B vs A' {float(np.median([d['B_vs_Ap'] for d in q]))!r}; min {min(d['B_vs_Ap'] for d in q)!r}; "
                f"max {max(d['B_vs_Ap'] for d in q)!r}; directions with B total <= 2B {sum(d['B_sum_ok'] for d in q)}/{len(q)}; "
                f"with each stage <= B {sum(d['B_each_ok'] for d in q)}/{len(q)}; A' total <= 2B {sum(d['Ap_sum_ok'] for d in q)}/{len(q)} ({VERDICT})")
    json.dump(dict(theta0=rows, check=dict(logdet_Ap=ldAp, logdet_B=ldB, B_minus_Ap=ldB-ldAp), p4=p4, B2=B2),
              open(os.path.join(RES, 'partB.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['schemeA', 'schemeD-redesign', 'schemeD-load', 'combine'])
    ap.add_argument('--workers', type=int, default=4)
    A_ = ap.parse_args()
    sys.stdout = dr.Tee(os.path.join(LOGS, f'partB_{A_.stage}.log'))
    print(f"=== two_stage.py {A_.stage} started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round5_CN.md v3.2; "
          f"B = {B!r}; 2B = {B2!r}")
    t0 = time.perf_counter()
    {'schemeA': lambda: schemeA(print, A_.workers), 'schemeD-redesign': lambda: schemeD_redesign(print, A_.workers),
     'schemeD-load': lambda: schemeD_load(print), 'combine': lambda: combine(print)}[A_.stage]()
    print(f"=== two_stage.py {A_.stage} done in {time.perf_counter()-t0:.1f} s")
