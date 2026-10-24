"""Round 5, Part 0 (round-4 maintenance; spec reports_CN/plan_round5_CN.md v3.2, §2 item 5): verification of 0b and the
numbers for 0c. No DP, no Monte Carlo.

0b  round4/part_c.py::load_table now reads the committed round4/results/table_<case>.npz. The deterministic C1 and C3 stages of
    round 4 are re-run in-process from the npz (round4/results/tables/, the uncommitted memmaps, is absent in this clone) and
    their products are compared with the committed round-4 files: c1.json (every field except the wall-clock eval_s), the six
    J_*.npy (bytes), c3_table.md (bytes), shape_*.json (bytes); the two figures are compared for information only. The committed
    round-4 files are then written back unchanged (the round-4 archive and its logs are not modified).
0c  information split by branch, w_k = sum_i m_{k,i} (c^ev + c^no) (dr.cfun returns the two terms): the same forward pass is run
    twice with cfun's output masked to one term (the mass recursion does not involve c, so m_k and E[N] are unchanged); the send
    branch includes the forced first sample (w_1 = 1). Uniform input: constant-Delta SoD (accepted operator, as in C1), S table,
    weighted table; P5514 logged as well. Also (S - 1)/(E[N] - 1), the mean send probability (E[N] - 1)/(K - 1) and c/q of the
    single step alpha = -2, beta = 30.

Usage (from this directory): python3 part0_verify.py   -> logs/part0_verify.log, results/part0_verify.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, glob, time
HERE = os.path.dirname(os.path.abspath(__file__)); R4 = os.path.join(os.path.dirname(HERE), 'round4')
sys.path.insert(0, R4)
import numpy as np
import evaluator_window as ew
import evaluator_fast as ef
import part_c as pc
dr = ew.dr
s, K, D = ew.s, ew.K, ew.D
LOGS = os.path.join(HERE, 'logs'); RES = os.path.join(HERE, 'results')
os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)

PRODUCTS = (['results/c1.json'] + [f'results/J_{c}.npy' for c in ('U_SoD', 'U_S', 'U_W', 'P5514_SoD', 'P5514_S', 'P5514_W')]
            + ['c3_table.md'] + [f'results/shape_{c}.json' for c in ('U_S', 'U_W', 'P5514_S', 'P5514_W')])
FIGS = ['fig_window_heatmap.png', 'fig_sendprob_trA.png']


def verify_0b(out):
    tdir = os.path.join(R4, 'results', 'tables')
    nmm = len(glob.glob(os.path.join(tdir, '*'))) if os.path.isdir(tdir) else 0
    out(f"0b round4/results/tables/ (uncommitted memmaps) holds {nmm} files; the tables can only come from results/table_<case>.npz")
    orig = {p: open(os.path.join(R4, p), 'rb').read() for p in PRODUCTS+FIGS}
    c1_old = json.loads(orig['results/c1.json'])
    res = {}
    try:
        t0 = time.perf_counter(); pc.c1(out); t1 = time.perf_counter()-t0
        t0 = time.perf_counter(); pc.c3(out); t3 = time.perf_counter()-t0
        out(f"0b re-run of round-4 C1 ({t1:.1f} s) and C3 ({t3:.1f} s) from the npz done")
        new = {p: open(os.path.join(R4, p), 'rb').read() for p in PRODUCTS+FIGS}
    finally:
        for p, b in orig.items(): open(os.path.join(R4, p), 'wb').write(b)        # the committed round-4 files, unchanged
    c1_new = json.loads(new['results/c1.json'])
    diff = []
    for case in c1_old:
        for k in c1_old[case]:
            if k == 'eval_s': continue
            if c1_new[case].get(k) != c1_old[case][k]: diff.append((case, k, c1_old[case][k], c1_new[case].get(k)))
        if set(c1_new[case]) != set(c1_old[case]): diff.append((case, 'keys', sorted(c1_old[case]), sorted(c1_new[case])))
    res['c1_json_fields_equal_except_eval_s'] = (not diff) and set(c1_new) == set(c1_old)
    out(f"0b results/c1.json: every field of every case equal except eval_s (wall clock): {res['c1_json_fields_equal_except_eval_s']}"
        + (f"; differences {diff!r}" if diff else ''))
    for case in c1_old:
        for k in ('EN', 'S', 'logdet', 'dmD', 'mE'):
            out(f"0b C1 {case} {k} committed {c1_old[case][k]!r} re-run {c1_new[case][k]!r} equal {c1_old[case][k] == c1_new[case][k]}")
    for p in PRODUCTS[1:]:
        same = new[p] == orig[p]; res[f'bytes_equal:{p}'] = same
        out(f"0b {p}: re-run bytes identical to the committed file: {same}")
    for p in FIGS:
        out(f"0b {p} (information only): bytes identical {new[p] == orig[p]}")
    res['c3_table_bitwise'] = res['bytes_equal:c3_table.md']
    res['pass'] = bool(res['c1_json_fields_equal_except_eval_s'] and all(v for k, v in res.items() if k.startswith('bytes_equal:')))
    out(f"0b C3 table reproduced bit for bit: {res['c3_table_bitwise']}; Part 0b verification {'PASS' if res['pass'] else 'FAIL'}")
    out("0b c3_table.md as re-run:")
    for ln in new['c3_table.md'].decode().splitlines(): out(f"0b   {ln}")
    return res


def _masked(which, full):
    def f(al, be):
        cev, cno = full(al, be)
        return (cev, np.zeros_like(cno)) if which == 'ev' else (np.zeros_like(cev), cno)
    return f

def split_run(fn):
    """fn() -> forward record with W; run with c masked to the send term, then to the non-send term."""
    full = dr.cfun; outs = {}
    try:
        for which in ('ev', 'no'):
            dr.cfun = _masked(which, full); outs[which] = fn()
    finally:
        dr.cfun = full
    return outs

def branch_0c(out):
    ew._warmup(); res = {}
    c1 = json.load(open(os.path.join(R4, 'results', 'c1.json')))
    for name in ew.INPUTS:
        xk, Hk, z, dz, a, b = ew.design_grid(name)
        rows = [('SoD', lambda: ef.run_window(xk, ew.N_GRID, 8*D))]
        for obj in ('S', 'W'):
            case = f'{name}/{obj}'
            with np.load(pc.table_npz(case)) as f: codes = f['codes']
            tab = ew.CodeTable(codes, a[1:], (b-a+1)[1:])
            rows.append((obj, (lambda tab=tab: ew.forward(z, dz, s, xk, a, b, tab))))
        for lab, fn in rows:
            o = split_run(fn); key = f'{name}/{lab}'; ref = c1[key]
            S_ev = float(o['ev']['W'].sum()); S_no = float(o['no']['W'].sum()-1.0)       # W[0] = 1 (forced first sample) in both
            EN = float(o['ev']['EN']); assert EN == float(o['no']['EN'])
            d = dict(S_send=S_ev, S_nosend=S_no, S_sum=S_ev+S_no, S_c1=ref['S'], sum_rel_dev=abs(S_ev+S_no-ref['S'])/ref['S'],
                     EN=EN, EN_c1=ref['EN'], EN_equal_c1=EN == ref['EN'],
                     ratio_S1_EN1=(ref['S']-1.0)/(ref['EN']-1.0), mean_send_prob=(ref['EN']-1.0)/(K-1),
                     send_info_per_send=(S_ev-1.0)/(ref['EN']-1.0))
            res[key] = d
            for k, v in d.items(): out(f"0c {key} {k} {v!r}")
    for name in ew.INPUTS:
        for obj in ('S', 'W'):
            d0, d1 = res[f'{name}/SoD'], res[f'{name}/{obj}']
            out(f"0c {name}/{obj} vs {name}/SoD: send branch {d0['S_send']!r} -> {d1['S_send']!r} (change {d1['S_send']-d0['S_send']!r}); "
                f"non-send branch {d0['S_nosend']!r} -> {d1['S_nosend']!r} (change {d1['S_nosend']-d0['S_nosend']!r})")
    cev, cno = dr.cfun(np.array([-2.0]), np.array([30.0])); c = float(cev[0]+cno[0]); q = float(1-(dr.ndtr(30.0)-dr.ndtr(-2.0)))
    res['single_step'] = dict(alpha=-2.0, beta=30.0, c=c, q=q, c_over_q=c/q, c_ev=float(cev[0]), c_no=float(cno[0]))
    for k, v in res['single_step'].items(): out(f"0c single step (illustration only) {k} {v!r}")
    return res


if __name__ == '__main__':
    sys.stdout = dr.Tee(os.path.join(LOGS, 'part0_verify.log'))
    import numba, scipy, matplotlib
    print(f"=== round5/part0_verify.py started {time.strftime('%Y-%m-%d %H:%M:%S')}; python {sys.version.split()[0]}, numpy {np.__version__}, "
          f"scipy {scipy.__version__}, numba {numba.__version__}, matplotlib {matplotlib.__version__}; B = {ew.B_BUDGET!r}")
    t0 = time.perf_counter()
    r0b = verify_0b(print)
    r0c = branch_0c(print)
    json.dump(dict(part0b=r0b, part0c=r0c), open(os.path.join(RES, 'part0_verify.json'), 'w'), indent=1)
    print(f"=== round5/part0_verify.py done in {time.perf_counter()-t0:.1f} s")
