"""Speed-up Step 4: regression of the window path against the accepted results (spec reports_CN/plan_speedup_CN.md v3.1, §5).

  1. the 12 accepted version-B evaluations (U, D, E, N x n = 6, 12, 24, margin 8 Delta) on the window path, compared with the
     stored accepted outputs at full precision (results/rec_B_*_m8.json, J_B_*, w_B_*), tolerances frozen in config_fast.json:
     E[N] (P and Phat), S, log det J: |fast - ref|/|ref| <= 1e-12; each of the 81 J elements |Delta| <= 1e-12 (|ref| + max|J_ref|);
     each w_k |Delta| <= 1e-12 (|w_k,ref| + 1); mass: the window path's own per-step max |sum m - 1| <= 1e-10;
  2. unit tests (b) and (c) on the window path, version B only (A is not on the window path): (b) Phat at h = 1/12, 1/24 against
     the accepted density_recursion_v1.single_step (relative <= 1e-12) and |B - 0.5| <= 1e-6; (c) U at n = 6, 12, 24 rerun:
     mass <= 1e-10, min m >= 0, E[N] and log det against the accepted records (relative <= 1e-12);
  3. tail check on the window path: B, n = 12, margin 16 Delta against 8 Delta, the quantities of item 1 with 1e-12 -> 1e-11;
     the accepted version's own 16-vs-8 differences are logged next to it (S5);
  4. assertion count over 12 x 4999 steps (must be 0); node counts, initialisation and later updates separately.
Any assertion failure or failed check stops the script (exit 1); nothing later may use the window path then.
Writes logs/speedup_regression.log (one line per number), results/fast/ (records, J, w_k, per-step CSV). Usage: python3 regression_fast.py"""
import os, sys, json
import numpy as np
import density_recursion_v1 as dr
import evaluator_fast as ef

RES = dr.RES; FD = os.path.join(RES, 'fast'); os.makedirs(FD, exist_ok=True)
INPUTS = ef.INPUTS; GRIDS = ef.GRIDS; POS = ['EN_Phat', 'EN_P', 'S', 'logdet']
FAILS = []


def check(tag, key, val, tol, ok=None):
    ok = (val <= tol) if ok is None else ok
    print(f"REG {tag} {key} {val!r} tol {tol!r} {'pass' if ok else 'FAIL'}")
    if not ok: FAILS.append(f"{tag} {key}")
    return ok

def compare(tag, rec, J, w, rrec, rJ, rw, f):
    """the three tolerance classes of spec §5 item 1 with factor f; returns the largest observed value per class."""
    worst = {}
    for q in POS:
        rel = float(abs(rec[q]-rrec[q])/abs(rrec[q]))
        print(f"REG {tag} {q} fast {float(rec[q])!r} ref {float(rrec[q])!r}")
        check(tag, f'{q}_reldiff', rel, f); worst['positive'] = max(worst.get('positive', 0.0), rel)
    jr = np.abs(J-rJ)/(np.abs(rJ)+np.abs(rJ).max())
    print(f"REG {tag} J max|Delta| {float(np.abs(J-rJ).max())!r}; elements over tolerance {int((jr > f).sum())} of 81")
    check(tag, 'J_max_|Delta|/(|ref|+max|J_ref|)', float(jr.max()), f); worst['J'] = float(jr.max())
    wr = np.abs(w-rw)/(np.abs(rw)+1)
    print(f"REG {tag} w_k max|Delta| {float(np.abs(w-rw).max())!r}; values over tolerance {int((wr > f).sum())} of {w.size}")
    check(tag, 'w_max_|Delta|/(|w_ref|+1)', float(wr.max()), f); worst['w'] = float(wr.max())
    return worst


def run_fast(name, n, mm):
    tag = f"B/{name}/n={n}/{mm}Delta"
    try:
        rec, J, r = ef.evaluate_fast(name, n, mm)
    except ef.AssertionStop as e:
        print(f"REG {tag} STOP: {e}"); print("REGRESSION STOPPED (assertion); window path not used further"); sys.exit(1)
    b = f"B_{name}_n{n}_m{mm}"
    np.save(os.path.join(FD, f'J_{b}.npy'), J); np.save(os.path.join(FD, f'w_{b}.npy'), r['W'])
    np.save(os.path.join(FD, f'mass_{b}.npy'), r['mass']); ef.write_steps(r, os.path.join(FD, f'steps_{b}.csv'))
    sm = ef.step_summary(r); rec['steps'] = sm
    json.dump({k: v for k, v in rec.items() if k != 'lam'}, open(os.path.join(FD, f'rec_{b}.json'), 'w'), indent=1)
    for k, v in sm.items(): print(f"REG {tag} {k} {v!r}")
    check(tag, 'mass_dev_max', rec['mass_dev_max'], 1e-10)
    check(tag, 'assertion_fail_count', sm['assert_fail_count'], 0)
    return tag, rec, J, r['W'], sm

def ref(name, n, mm):
    b = f"B_{name}_n{n}_m{mm}"
    return (json.load(open(os.path.join(RES, f'rec_{b}.json'))), np.load(os.path.join(RES, f'J_{b}.npy')),
            np.load(os.path.join(RES, f'w_{b}.npy')))


if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(dr.LOGS, 'speedup_regression.log'))
    print(f"=== regression_fast.py (speed-up Step 4); spec {ef.CFGF['spec']}; predictions {ef.CFGF['predictions_commit']}; "
          f"R = {ef.R/ef.s} sigma; tolerances {json.dumps(ef.CFGF['tolerances'], ensure_ascii=False)}")
    ef._warmup()
    worst = {}; summ = {}; nfail_assert = 0
    # 1. the 12 accepted evaluations
    for name in INPUTS:
        for n in GRIDS:
            tag, rec, J, w, sm = run_fast(name, n, 8)
            wv = compare(tag, rec, J, w, *ref(name, n, 8), 1e-12)
            for k, v in wv.items(): worst[k] = max(worst.get(k, 0.0), v)
            worst['mass'] = max(worst.get('mass', 0.0), rec['mass_dev_max'])
            summ[(name, n)] = sm; nfail_assert += sm['assert_fail_count']
            if FAILS: print(f"REGRESSION FAIL: {FAILS}; STOP"); sys.exit(1)
    for k, v in worst.items(): print(f"REG-WORST 12 evaluations, class {k}: largest observed {v!r}")
    print(f"REG-ASSERT assertion failures over {len(summ)} x {ef.K-1} steps: {nfail_assert}")
    for n in GRIDS:
        ss = [summ[(nm, n)] for nm in INPUTS]
        print(f"REG-NODES n={n}: initialisation non-zero nodes min {min(x['init_nonzero_nodes'] for x in ss)} max "
              f"{max(x['init_nonzero_nodes'] for x in ss)} (support extent max {max(x['init_support_nodes'] for x in ss)}); "
              f"later updates non-zero nodes min {min(x['update_nonzero_nodes_min'] for x in ss)} max "
              f"{max(x['update_nonzero_nodes_max'] for x in ss)} (support extent max {max(x['update_support_nodes_max'] for x in ss)}); "
              f"window nodes {min(x['window_nodes_min'] for x in ss)}-{max(x['window_nodes_max'] for x in ss)}")
    # 2. unit tests (b) and (c) on the window path (version B)
    for h, lab in ((1/12, '1/12'), (1/24, '1/24')):
        vf = ef.single_step_window(h); va = dr.single_step('B', h)
        print(f"REG (b) h={lab}: B window {vf!r}, accepted {va!r}")
        check(f"(b)/h={lab}", 'reldiff_vs_accepted', abs(vf-va)/abs(va), 1e-12)
        check(f"(b)/h={lab}", '|B-0.5|', abs(vf-0.5), 1e-6)
    for n in GRIDS:
        rec, J, r = ef.evaluate_fast('U', n, 8); rr = ref('U', n, 8)[0]; tag = f"(c)/B/U/n={n}"
        check(tag, 'mass_dev_max', rec['mass_dev_max'], 1e-10)
        check(tag, 'min_m_nonneg', rec['min_m'], 0.0, ok=rec['min_m'] >= 0)
        print(f"REG {tag} min_m {rec['min_m']!r}")
        for q in ('EN', 'logdet'): check(tag, f'{q}_reldiff_vs_accepted', abs(rec[q]-rr[q])/abs(rr[q]), 1e-12)
        same = np.array_equal(r['W'], np.load(os.path.join(FD, f'w_B_U_n{n}_m8.npy')))
        print(f"REG {tag} rerun identical to the regression run (w_k bitwise): {same}")
    # 3. tail check on the window path, and the accepted version's own tail differences for comparison (S5)
    tw = {}
    for name in INPUTS:
        tag, rec16, J16, w16, sm16 = run_fast(name, 12, 16)
        nfail_assert += sm16['assert_fail_count']
        r8 = json.load(open(os.path.join(FD, f'rec_B_{name}_n12_m8.json')))
        J8 = np.load(os.path.join(FD, f'J_B_{name}_n12_m8.npy')); w8 = np.load(os.path.join(FD, f'w_B_{name}_n12_m8.npy'))
        wv = compare(f"tail/{tag}", rec16, J16, w16, r8, J8, w8, 1e-11)
        for k, v in wv.items(): tw[k] = max(tw.get(k, 0.0), v)
        a8, aJ8, aw8 = ref(name, 12, 8); a16, aJ16, aw16 = ref(name, 12, 16)
        acc = dict(positive=float(max(abs(a16[q]-a8[q])/abs(a8[q]) for q in POS)),
                   J=float((np.abs(aJ16-aJ8)/(np.abs(aJ8)+np.abs(aJ8).max())).max()), w=float((np.abs(aw16-aw8)/(np.abs(aw8)+1)).max()))
        for k in ('positive', 'J', 'w'):
            print(f"REG tail/{name} class {k}: window path {wv[k]!r}, accepted version {acc[k]!r}")
        if FAILS: print(f"REGRESSION FAIL: {FAILS}; STOP"); sys.exit(1)
    for k, v in tw.items(): print(f"REG-WORST tail check (16 vs 8 Delta, window path), class {k}: largest observed {v!r}")
    print(f"REG-ASSERT assertion failures including the 4 tail-check runs: {nfail_assert}")
    if FAILS or nfail_assert: print(f"REGRESSION FAIL: {FAILS}; STOP"); sys.exit(1)
    print("REGRESSION PASS: all checks within the frozen tolerances, assertion count 0")
