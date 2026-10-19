"""Step 3: all evaluations of the acceptance run (spec reports_CN/plan_evaluator_CN.md v4.1, §6).

  phase 1  3 versions x 4 inputs x 3 grids (n = 6, 12, 24; margin 8 Delta) = 36 evaluations, then the tail check
           (version B, n = 12, margin 16 Delta, all four inputs: the relative quantities need U at 16 Delta).
           Serial, one process, timed (import and JIT excluded). U first at every (version, grid, margin): Delta m_D,
           lambda_i, m_E of D, E, N are taken against U at the same version, grid and margin.
  phase 2  c invariants on every node at every step (untimed, 4 worker processes): one pass per (c formula, input,
           grid); A and B share grid and c formula (mc83.cfun), so their pass is the same.
  phase 3  §6 trigger check on the absolute quantities (log det J, E[N], S) of every version and input:
           d1 = Q(6)-Q(12), d2 = Q(12)-Q(24), p_obs = log2|d1/d2|; trigger iff |p_obs - p0| > 0.5 (p0 = 1 for v0, A;
           2 for B) or sign(d1) != sign(d2); |d2| <= 1e-8 max(1, |Q(24)|) is "order undeterminable", never a trigger.
           The check is logged; n = 48 is NOT run here automatically.
  --n48 E,N   run version B at n = 48 for the listed inputs and for U (grid-switch rule, spec §6), only after a trigger.

Every evaluation is written to results/ as soon as it finishes (record JSON, J, w_k, per-step mass) and skipped when
rerun, so the script can be restarted after the sandbox kills it.  Log: logs/run_all.log (appended), one line per number.
Usage (from this directory):  nohup python3 run_all.py > logs/run_all.stdout 2>&1 &
"""
import os, sys, json, time, argparse
import numpy as np
import density_recursion_v1 as dr

INPUTS = ['U', 'D', 'E', 'N']; GRIDS = dr.CFG['grids']['n']; P0 = {'v0': 1, 'A': 1, 'B': 2}
ABS_Q = ['logdet', 'EN', 'S']


def rec_path(ver, name, n, mm): return os.path.join(dr.RES, f'rec_{ver}_{name}_n{n}_m{mm}.json')

def one(ver, name, n, mm, JU):
    """evaluate (or load) one case; returns (record, J)."""
    p = rec_path(ver, name, n, mm)
    if os.path.exists(p):
        rec = json.load(open(p)); return rec, np.load(os.path.join(dr.RES, f"J_{ver}_{name}_n{n}_m{mm}.npy"))
    rec, J, r = dr.evaluate(ver, name, n, mm)
    dr.relative(rec, J, J if name == 'U' else JU)
    if name == 'U':                                       # identities, not comparisons (spec §7): logged as checks
        rec['identity_dmD'] = rec.pop('dmD'); rec['identity_lam_maxdev'] = float(np.abs(np.array(rec.pop('lam'))-1).max())
        rec.pop('mE')
    dr.save(rec, J, r)
    dr.log_record(rec)
    for k in ('identity_dmD', 'identity_lam_maxdev'):
        if k in rec: print(f"EV {dr.tag(rec)} {k} {rec[k]!r}")
    json.dump(rec, open(p, 'w'), indent=1)
    return rec, J

def phase1(cases):
    for ver, n, mm in cases:
        JU = None
        for name in INPUTS:
            rec, J = one(ver, name, n, mm, JU)
            if name == 'U': JU = J

def _inv_worker(args):
    ver, name, n, mm = args
    return args, dr.c_invariants(ver, name, n, mm)

def phase2(cases, names=INPUTS):
    """c invariants; one pass per (formula, input, grid, margin), result stored in every record that uses it."""
    todo = {}
    for ver, n, mm in cases:
        for name in names:
            key = ('v0' if ver == 'v0' else 'AB', name, n, mm)
            todo.setdefault(key, []).append((ver, name, n, mm))
    jobs = []
    for key, users in todo.items():
        recs = [json.load(open(rec_path(*u))) for u in users]
        if all('inv_c' in r for r in recs): continue
        jobs.append(users[0])
    if jobs:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(4) as pl: res = pl.map(_inv_worker, jobs, chunksize=1)
        done = {('v0' if a[0] == 'v0' else 'AB', a[1], a[2], a[3]): v for a, v in res}
    else:
        done = {}
    for key, users in todo.items():
        for u in users:
            p = rec_path(*u); rec = json.load(open(p))
            if 'inv_c' in rec: continue
            rec.update(done[key]); rec['inv_c_pass_shared_with'] = [f"{v}/{nm}/n={nn}/{m}Delta" for v, nm, nn, m in users]
            json.dump(rec, open(p, 'w'), indent=1)
            for k in ('c_min', 'c_max', 'c_nan', 'c_nodes', 'tail_nodes', 'tail_c_min', 'tail_c_max', 'tail_nan', 'inv_c'):
                print(f"EV {dr.tag(rec)} {k} {rec[k]!r}")

def phase3():
    """§6 trigger check on the absolute quantities (margin 8 Delta, n = 6, 12, 24)."""
    trig = []
    print("TRIGGER-CHECK columns: version input quantity Q6 Q12 Q24 d1 d2 p_obs p0 status")
    for ver in dr.VERSIONS:
        for name in INPUTS:
            R = {n: json.load(open(rec_path(ver, name, n, 8))) for n in GRIDS}
            for q in ABS_Q:
                Q = [R[n][q] for n in GRIDS]; d1 = Q[0]-Q[1]; d2 = Q[1]-Q[2]
                if abs(d2) <= 1e-8*max(1.0, abs(Q[2])):
                    pobs = None; status = 'order_undeterminable(roundoff_d2)'
                else:
                    pobs = float(np.log2(abs(d1/d2))) if d1 != 0 else float('-inf')
                    if np.sign(d1) != np.sign(d2): status = 'TRIGGER(sign_change)'
                    elif abs(pobs-P0[ver]) > 0.5: status = 'TRIGGER(p_obs_off_by_>0.5)'
                    else: status = 'ok'
                if status.startswith('TRIGGER'): trig.append((ver, name, q, status))
                print(f"TRIGGER-CHECK {ver} {name} {q} {Q[0]!r} {Q[1]!r} {Q[2]!r} {d1!r} {d2!r} "
                      f"{'n/a' if pobs is None else repr(pobs)} {P0[ver]} {status}")
    print(f"TRIGGER-SUMMARY {len(trig)} trigger(s): " + ('; '.join(f'{v}/{nm}/{q}: {st}' for v, nm, q, st in trig) or 'none'))
    return trig


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--n48', default='')
    args = ap.parse_args()
    os.makedirs(dr.LOGS, exist_ok=True); os.makedirs(dr.RES, exist_ok=True)
    sys.stdout = dr.Tee(os.path.join(dr.LOGS, 'run_all.log'), 'a')
    t_start = time.time()
    print(f"=== run_all.py started {time.strftime('%Y-%m-%d %H:%M:%S')}; config {dr.CFG['spec']}; hardware {dr.CFG['hardware']}; "
          f"args {vars(args)}")
    dr._warmup()
    if args.n48:
        names = [x for x in args.n48.split(',') if x]
        assert all(x in INPUTS and x != 'U' for x in names)
        JU = one('B', 'U', 48, 8, None)[1]
        for name in names: one('B', name, 48, 8, JU)
        phase2([('B', 48, 8)], ['U']+names)
        print(f"n=48 (version B) run for U and {names} (grid-switch rule, spec §6)")
    else:
        cases = [(v, n, 8) for v in dr.VERSIONS for n in GRIDS] + [('B', 12, 16)]
        phase1(cases); phase2(cases); phase3()
    el = time.time()-t_start
    print(f"=== run_all.py finished {time.strftime('%Y-%m-%d %H:%M:%S')}, wall {el:.1f} s")
