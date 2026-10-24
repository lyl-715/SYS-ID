"""Unit test of the lambda scan in trigger_dp.Scan.scan with a synthetic oracle (round 5 Part 0a; spec
reports_CN/plan_round5_CN.md v3.2, §2 item 5): no DP, no evaluator call, no file outside a temporary directory.

The oracle replaces Scan.run: E[N](lambda) is a given non-increasing function of lambda/gbar, feasible = E[N] <= B, F = E[N]
(more sends, more information), the run cap of 40 is enforced as in Scan.run. Cases, each for gbar = 1 (objective S) and for
gbar = 0.0215873 (the uniform weighted objective's mean gamma):
  (a) feasible only from lambda/gbar = 90: the doubling must reach the cap 100 gbar, find it feasible and go on to the downward
      scan and bisection;
  (b) feasible only from lambda/gbar = 150: the doubling must evaluate 5, 10, 20, 40, 80 and the cap 100 gbar (no value above it),
      then stop with status 'no feasible policy in the lambda range';
  (c) feasible only from lambda/gbar = 100 (step oracle at the cap, 1e-9 below it to absorb the rounding of (100 gbar)/gbar):
      the cap itself is the only feasible value of the doubling and the final policy;
  (d) feasible at lambda_0 = 5 gbar: no doubling (the path every round-4 case took).

Usage (from this directory): python3 test_trigger_dp.py [MODULE_FILE]   # default: ./trigger_dp.py; log -> logs/test_trigger_dp.log
A MODULE_FILE argument runs the same cases against another copy of the module (used once on the pre-fix revision, to show
that the test exercises the cap).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, shutil, tempfile, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def load(path):
    spec = importlib.util.spec_from_file_location('trigger_dp_under_test', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def oracle_scan(td, gbar, en_of, tmp, log):
    """a Scan object whose run() is the synthetic oracle en_of(lambda/gbar) -> E[N]."""
    sc = object.__new__(td.Scan)
    sc.case = 'SYN/oracle'; sc.jpath = os.path.join(tmp, 'scan_syn.json'); sc.log = log; sc.gbar = gbar
    sc.state = dict(case=sc.case, runs=[], status='running'); sc.memo = {}
    def run(lam, phase):
        if lam in sc.memo: return sc.memo[lam]
        n = len(sc.state['runs'])
        if n >= td.MAX_RUNS: raise td.RunCap()
        EN = float(en_of(lam/gbar))
        r = dict(run=n+1, phase=phase, lam=lam, lam_over_gbar=lam/gbar, EN=EN, N=EN-1.0, S=EN, F=EN, dmD=0.0, dual=0.0,
                 feasible=bool(EN <= td.B), slack=td.B-EN)
        sc.state['runs'].append(r); sc.memo[lam] = r
        return r
    sc.run = run
    return sc

def cases(B):
    return {
        'a': (lambda lu: 1.0+(B-1.0)*90.0/lu, 'feasible from lambda/gbar = 90'),
        'b': (lambda lu: 1.0+(B-1.0)*150.0/lu, 'feasible from lambda/gbar = 150 (beyond the cap)'),
        'c': (lambda lu: B-0.1 if lu >= 100.0-1e-9 else B+1.0, 'feasible from lambda/gbar = 100 (step at the cap; (100 gbar)/gbar may round below 100)'),
        'd': (lambda lu: 1.0+(B-1.0)*3.4/lu, 'feasible at lambda_0 = 5 gbar'),
    }

def run_all(td, log):
    ok_all = True; tmp = tempfile.mkdtemp()
    try:
        for gbar in (1.0, 0.0215873):
            cap = td.LAM_CAP*gbar if hasattr(td, 'LAM_CAP') else 100.0*gbar
            for key, (en_of, desc) in cases(td.B).items():
                sc = oracle_scan(td, gbar, en_of, tmp, lambda m: None)
                st = sc.scan(); runs = st['runs']
                lams = [r['lam'] for r in runs]; dbl = [r['lam'] for r in runs if r['phase'] in ('start', 'double')]
                fin = [r for r in runs if r['run'] == st.get('final_run')]
                checks = {}
                checks['no lambda above the cap'] = max(lams) <= cap
                if key in ('a', 'b', 'c'):
                    exp = [5.0*gbar*2**i for i in range(5)]+[cap]
                    checks['doubling values 5, 10, 20, 40, 80, 100 gbar'] = dbl == exp
                    checks['cap evaluated'] = cap in lams
                if key == 'a':
                    checks["status 'done'"] = st['status'] == 'done'
                    checks['final lambda/gbar in [90, 100]'] = bool(fin) and 90.0-1e-9 <= fin[0]['lam']/gbar <= 100.0+1e-9
                    checks['downward scan from the cap ran (80 gbar again, memoised)'] = any(r['phase'] == 'scan' for r in runs) or (0.8*cap in sc.memo)
                if key == 'b':
                    checks["status 'no feasible policy in the lambda range'"] = st['status'] == 'no feasible policy in the lambda range'
                    checks['six runs'] = len(runs) == 6
                if key == 'c':
                    checks["status 'done'"] = st['status'] == 'done'
                    checks['final lambda = the cap'] = bool(fin) and fin[0]['lam'] == cap
                if key == 'd':
                    checks['no doubling'] = dbl == [5.0*gbar]
                    checks["status 'done'"] = st['status'] == 'done'
                ok = all(checks.values()); ok_all &= ok
                log(f"TEST gbar={gbar!r} case ({key}) {desc}: runs {len(runs)}, status {st['status']!r}, doubling/start lambda/gbar "
                    f"{[round(v/gbar, 12) for v in dbl]}, final lambda/gbar {fin[0]['lam']/gbar if fin else None!r} -> {'PASS' if ok else 'FAIL'}")
                for k, v in checks.items(): log(f"TEST gbar={gbar!r} case ({key})   {k}: {v}")
    finally:
        shutil.rmtree(tmp)
    return ok_all


if __name__ == '__main__':
    path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, 'trigger_dp.py')
    td = load(path)
    lf = open(os.path.join(HERE, 'logs', 'test_trigger_dp.log'), 'a')
    def log(m): print(m); lf.write(m+'\n'); lf.flush()
    log(f"=== test_trigger_dp.py (synthetic oracle, no DP) against {os.path.relpath(path, HERE)}; B = {td.B!r}; "
        f"cap {getattr(td, 'LAM_CAP', 'not defined (pre-fix revision: 100 gbar hard-coded)')}")
    ok = run_all(td, log)
    log(f"TEST {'ALL PASS' if ok else 'FAILURES'}")
    sys.exit(0 if ok else 1)
