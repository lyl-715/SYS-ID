"""Round 7, Part 0 verification (spec reports_CN/plan_round7_CN.md v3.1, frozen; §3 item 4): the round-6 engineering fixes
change no number.

  1. round6/margin_design.py final  -- recomputes every row of round6/results/margin_final.json from the committed npz
     tables only (checks git ls-files + unmodified against HEAD) and compares field by field (wall clock excluded).
  2. round6/robustness_margin.py sweep with the per-point checkpoint, outputs redirected to results/runs/part0_verify/A:
     first interrupted after 20 points (--max-new 20), then resumed; the assembled archive and J npz are compared with the
     committed round6/results/archive_partA.jsonl, J_partA.npz, partA_sweep.json (every field except wall_s / runtime_s;
     81 J elements bitwise; same line order).
  3. round6/refine_check.py with the per-evaluation checkpoint, outputs to results/runs/part0_verify/B: interrupted after
     5 evaluations, resumed; compared with the committed round6/results/partB_refine.json (every field except wall clock).

Usage (from this directory): python3 part0_verify.py   -> logs/part0_verify.log, results/part0_verify.json
"""
import os, sys, json, time, subprocess, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE); R6 = os.path.join(EV, 'round6')
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results'); VD = os.path.join(RES, 'runs', 'part0_verify')
TIMING = {'wall_s', 'runtime_s', 'check_s'}
LOG = open(os.path.join(LOGS, 'part0_verify.log'), 'w')
def out(m): print(m); LOG.write(m+'\n'); LOG.flush()

def run(args, tag):
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable]+args, cwd=R6, capture_output=True, text=True)
    w = time.perf_counter()-t0
    out(f"P0-RUN {tag}: python3 {' '.join(args)} (cwd round6) exit {p.returncode}, {w:.1f} s")
    if p.returncode != 0: out(p.stdout[-3000:]+p.stderr[-3000:])
    return p, w

def same(a, b): return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

def strip(d): return {k: v for k, v in d.items() if k not in TIMING}

def main():
    if os.path.exists(VD): shutil.rmtree(VD)
    os.makedirs(VD); res = {}; walls = {}
    out(f"=== part0_verify.py started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round7_CN.md v3.1 §3 item 4")
    # 1. margin_design final from the committed npz
    p, walls['final'] = run(['margin_design.py', 'final'], 'margin_design final (committed npz)')
    mf = json.load(open(os.path.join(R6, 'results', 'margin_final_npz.json')))
    for t, c in mf['compare'].items():
        out(f"P0-1 {t} recomputed fields equal to margin_final.json: {sum(c.values())}/{len(c)}; unequal {[k for k, v in c.items() if not v]}")
    res['final'] = dict(pass_=bool(p.returncode == 0 and mf['pass_']), compare=mf['compare'])
    out(f"P0-1 margin_design.py final reproduces margin_final.json from committed files only: {'PASS' if res['final']['pass_'] else 'FAIL'}")
    # 2. robustness_margin sweep: interrupted, resumed, compared
    A = os.path.join(VD, 'A')
    p1, walls['sweep_part1'] = run(['robustness_margin.py', 'sweep', '--outdir', A, '--max-new', '20'], 'sweep interrupted after 20 points')
    n1 = sum(1 for _ in open(os.path.join(A, 'partA_sweep.ckpt.jsonl')))
    interrupted = not os.path.exists(os.path.join(A, 'archive_partA.jsonl'))
    out(f"P0-2 after the interrupted run: {n1} points in the checkpoint (>= 20 expected), archive not yet written: {interrupted}")
    p2, walls['sweep_part2'] = run(['robustness_margin.py', 'sweep', '--outdir', A], 'sweep resumed')
    resumed = [l for l in open(os.path.join(A, 'partA_sweep.log')) if l.startswith('A2-CKPT')]
    for l in resumed: out(f"P0-2 log: {l.strip()}")
    ref = [json.loads(l) for l in open(os.path.join(R6, 'results', 'archive_partA.jsonl'))]
    new = [json.loads(l) for l in open(os.path.join(A, 'archive_partA.jsonl'))]
    bad = [i for i, (x, y) in enumerate(zip(ref, new)) if not same(strip(x), strip(y)) or set(x) != set(y)]
    Jr = np.load(os.path.join(R6, 'results', 'J_partA.npz')); Jn = np.load(os.path.join(A, 'J_partA.npz'))
    jbad = [k for k in Jr.files if k not in Jn.files or not np.array_equal(Jr[k], Jn[k])] + [k for k in Jn.files if k not in Jr.files]
    sr = json.load(open(os.path.join(R6, 'results', 'partA_sweep.json'))); sn = json.load(open(os.path.join(A, 'partA_sweep.json')))
    sw_ok = same(strip(sr), strip(sn))
    ok2 = bool(p1.returncode == 0 and p2.returncode == 0 and len(ref) == len(new) == 405 and not bad and not jbad and sw_ok and interrupted and n1 >= 20)
    res['sweep'] = dict(points_after_interrupt=n1, lines=len(new), line_mismatch=bad, J_mismatch=jbad, partA_sweep_json_equal=sw_ok, pass_=ok2)
    out(f"P0-2 archive lines {len(new)} (committed {len(ref)}), same order; lines differing in any non-timing field: {bad}")
    out(f"P0-2 J_partA.npz: {len(Jn.files)} arrays (committed {len(Jr.files)}), arrays not bitwise equal: {jbad}")
    out(f"P0-2 partA_sweep.json equal apart from wall clock (n, n_ok, n_stop, regression block): {sw_ok}")
    out(f"P0-2 checkpointed sweep (interrupted + resumed) reproduces the committed Part A archive: {'PASS' if ok2 else 'FAIL'}")
    # 3. refine_check: interrupted, resumed, compared
    Bd = os.path.join(VD, 'B')
    q1, walls['refine_part1'] = run(['refine_check.py', '--outdir', Bd, '--max-new', '5'], 'refine_check interrupted after 5 evaluations')
    m1 = sum(1 for _ in open(os.path.join(Bd, 'partB_refine.ckpt.jsonl')))
    intr_b = not os.path.exists(os.path.join(Bd, 'partB_refine.json'))
    q2, walls['refine_part2'] = run(['refine_check.py', '--outdir', Bd], 'refine_check resumed')
    for l in open(os.path.join(Bd, 'partB_refine.log')):
        if l.startswith('B0-CKPT'): out(f"P0-3 log: {l.strip()}")
    br = json.load(open(os.path.join(R6, 'results', 'partB_refine.json'))); bn = json.load(open(os.path.join(Bd, 'partB_refine.json')))
    rbad = [i for i, (x, y) in enumerate(zip(br['rows'], bn['rows'])) if not same(strip(x), strip(y))]
    top = same(strip({k: v for k, v in br.items() if k != 'rows'}), strip({k: v for k, v in bn.items() if k != 'rows'}))
    ok3 = bool(q1.returncode == 0 and q2.returncode == 0 and intr_b and m1 >= 5 and len(br['rows']) == len(bn['rows']) and not rbad and top)
    res['refine'] = dict(evals_after_interrupt=m1, rows=len(bn['rows']), row_mismatch=rbad, other_fields_equal=top, pass_=ok3)
    out(f"P0-3 after the interrupted run: {m1} evaluations in the checkpoint, partB_refine.json not yet written: {intr_b}")
    out(f"P0-3 rows {len(bn['rows'])} (committed {len(br['rows'])}); rows differing in any non-timing field: {rbad}; other fields equal: {top}")
    out(f"P0-3 checkpointed refine_check (interrupted + resumed) reproduces the committed Part B record: {'PASS' if ok3 else 'FAIL'}")
    allok = res['final']['pass_'] and ok2 and ok3
    res['pass_'] = allok; res['wall_s'] = walls
    out(f"P0 wall clock of the verification runs: {walls}; total {sum(walls.values()):.1f} s")
    out(f"P0 Part 0 verification (no numerical change): {'PASS' if allok else 'FAIL'}")
    json.dump(res, open(os.path.join(RES, 'part0_verify.json'), 'w'), indent=1)
    return allok

if __name__ == '__main__':
    sys.exit(0 if main() else 1)
