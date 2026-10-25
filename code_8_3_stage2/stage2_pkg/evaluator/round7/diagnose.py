"""Round 7, Part B: quadrature diagnosis (spec reports_CN/plan_round7_CN.md v3.1, frozen; §1.2, §3 item 2). A diagnostic item,
not a pass/fail check. 固定设计、离散方案下.

Frozen point P1 +0.5 (theta_1 + 0.5 sigma), the m = 0.20 table (U_W_m020_r08). E[N] of the same frozen continuous policy with
three quadratures (DECISION B-1):
  (i)   the original sigma/48 grid: round 6 Part B's path itself (round6/refine_check.eval_table48 -- round 4's FineTable,
        nearest coarse node with ties -> smaller index; the odd nodes lie on the cell boundaries), recomputed and asserted
        equal to round6/results/partB_refine.json;
  (ii)  a sigma/48 grid whose nodes are shifted by h_48/2 (z_0 + (j + 1/2) h_48; coarse origin, cell boundaries and the
        physical policy untouched), new cell lookup (every node 1/4 cell from a boundary);
  (iii) sigma/72 (f = 3), new cell lookup.
Reported: the three values, (i) - (iii), (ii) - (iii). Reading (spec §1.2, B2): |(i) - (iii)| > 1 send and
|(ii) - (iii)| < 0.1 send -> "支持分界取值影响的解释"; otherwise "未获得支持／误差来源仍未定位". One diagnosis excludes no
other error source. Each evaluation is appended to results/partB_diagnose.jsonl as it completes (checkpoint; a re-run resumes).

Usage (from this directory): python3 diagnose.py   -> logs/partB_diagnose.log, results/partB_diagnose.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import evaluator_refine as er
import refine_check as rc                                  # round 6 Part B, unchanged
ew, dr, rb = er.ew, er.dr, er.rb
B = er.B
PID = 'P1|theta1|+|0.5'; TID = 'm020'
ARCH = os.path.join(er.RES, 'partB_diagnose.jsonl'); OUTJ = os.path.join(er.RES, 'partB_diagnose.json')
JOBS = {'i_sigma48': 'original sigma/48 (round 6 Part B path, FineTable, ties -> smaller index)',
        'ii_sigma48_shift': 'sigma/48 shifted by h_48/2 (new cell lookup)',
        'iii_sigma72': 'sigma/72 (f = 3, new cell lookup)'}

def _job(key):
    er.warmup()
    pt = [p for p in er.points27() if p['pid'] == PID][0]; tab = er.load_table(TID); xth = er.x_theta(pt['theta'])
    t0 = time.perf_counter()
    if key == 'i_sigma48':
        rec, J, r = rc.eval_table48(tab, xth); rec['route'] = 'general'; rec['f'] = 2; rec['shift'] = False
        rec['assert_fail'] = int((~r['assert_ok']).sum())
    elif key == 'ii_sigma48_shift': rec, J, r = er.evaluate(tab, 2, xth=xth, shift=True)
    else: rec, J, r = er.evaluate(tab, 3, xth=xth)
    d = er.record(rec, J); d.update(key=key, what=JOBS[key], wall_s=time.perf_counter()-t0)
    return d

def main(out):
    import multiprocessing as mp
    done = {}
    if os.path.exists(ARCH):
        for l in open(ARCH): d = json.loads(l); done[d['key']] = d
    todo = [k for k in JOBS if k not in done]
    out(f"B0 frozen point {PID} (theta_1 + 0.5 sigma), table {TID}; archive {os.path.relpath(ARCH, HERE)}: {len(done)} resumed, {len(todo)} to evaluate")
    t0 = time.perf_counter()
    if todo:
        with mp.get_context('fork').Pool(min(4, len(todo))) as pool, open(ARCH, 'a') as f:
            for d in pool.imap_unordered(_job, todo, chunksize=1):
                f.write(json.dumps(d, ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno()); done[d['key']] = d
    wall = time.perf_counter()-t0
    v = {k: done[k]['EN'] for k in JOBS}
    for k in JOBS:
        d = done[k]
        out(f"B1 {k} ({d['what']}): E[N] {d['EN']!r}; S {d['S']!r}; log det {d['logdet']!r}; G {d['G']}; route {d['route']}; "
            f"mass dev {d['mass_dev_max']!r}; min m {d['min_m']!r}; assertion failures {d['assert_fail']}; wall {d['wall_s']:.1f} s")
    # regression of (i) against round 6 Part B
    r6 = [r for r in json.load(open(os.path.join(er.R6, 'results', 'partB_refine.json')))['rows'] if r['point'] == PID][0]
    reg = dict(EN=v['i_sigma48'] == r6['EN48'], logdet=done['i_sigma48']['logdet'] == r6['logdet48'])
    out(f"B1-REG (i) vs round6/results/partB_refine.json {PID} (exact): {reg}; round 6 E[N]_48 {r6['EN48']!r}")
    A6 = [json.loads(l) for l in open(os.path.join(er.R6, 'results', 'archive_partA.jsonl'))]
    n24 = [r for r in A6 if r['table'] == TID and r['cls'] == 'P1' and r['sign'] > 0 and r['eps_target'] == 0.5][0]['EN']
    out(f"B1 context (not used in the reading): sigma/24 E[N] {n24!r} (round 6 archive)")
    d13 = v['i_sigma48']-v['iii_sigma72']; d23 = v['ii_sigma48_shift']-v['iii_sigma72']
    out(f"B1 (i) - (iii) = {d13!r} send")
    out(f"B1 (ii) - (iii) = {d23!r} send")
    supp = abs(d13) > 1.0 and abs(d23) < 0.1
    reading = '支持分界取值影响的解释' if supp else '未获得支持／误差来源仍未定位'
    out(f"B2 reading (diagnostic, not a pass/fail gate): |(i) - (iii)| > 1: {abs(d13) > 1.0}; |(ii) - (iii)| < 0.1: {abs(d23) < 0.1} -> {reading}; "
        f"one diagnosis excludes no other error source")
    res = dict(point=PID, table=TID, values=v, records={k: {kk: vv for kk, vv in done[k].items() if kk != 'J'} for k in JOBS},
               d_i_iii=d13, d_ii_iii=d23, reading=reading, supported=supp, regression_round6=reg, EN24_context=n24, wall_s=wall)
    json.dump(res, open(OUTJ, 'w'), indent=1, ensure_ascii=False)
    return res


if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(er.LOGS, 'partB_diagnose.log'), 'a')
    print(f"=== diagnose.py started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round7_CN.md v3.1 §1.2; B = {B!r}")
    t0 = time.perf_counter(); main(print); print(f"=== diagnose.py done in {time.perf_counter()-t0:.1f} s")
