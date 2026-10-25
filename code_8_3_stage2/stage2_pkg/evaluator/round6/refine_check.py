"""Round 6, Part B: sigma/48 grid check of the selected table only (spec reports_CN/plan_round6_CN.md v3.2, frozen; §2).
固定设计、离散方案下 -- the same frozen continuous policy evaluated on a finer reference grid; no fine-grid redesign, no bound.

Policy transport (round 4 §1.4, evaluator_window.FineTable, unchanged): fine node j looks up the nearest coarse node (ties ->
smaller index, j // 2 on the exactly nested grids), outside the step's stored range the boundary node of that range; physical
half-widths kept (grid counts doubled); window centred at the fine node's own r. Under mismatch the round-5 conventions are
carried to the fine grid (DECISION B-1): the frozen fine grid is dr.make_grid of the NOMINAL output at n = 48; support windows
windows88 centred at x(theta) on it; record and J from the accepted dr.evaluate (its input arrays and run_ab replaced for the
call). The sigma/48 SoD at theta is evaluator_fast.evaluate_fast(U, 48) with input arrays (x(theta), H) and its grid pinned to
the nominal fine grid (round 4 C2's reference SoD at theta_0). sigma/24 values are those of the Part A archive.

Points (spec §2): theta_0; the point with the largest E[N] within the 26-point selection set; the point with the lowest
Delta m_D within that set (main check points, used for the verdict); supplementary (not used): the largest-E[N] and the
lowest-Delta m_D points among all 80. A main point is 精度未定 if the budget verdict flips, |N_48 - N_24| > 0.05, or
min(|N_24 - B|, |N_48 - B|) <= 0.05 (0.05 is a decision threshold, not a certified error bound).
Path regression: the m = 0 table and the SoD at theta_0 through this path against round 4 C2 (round4/results/c2.json).

Usage (from this directory): python3 refine_check.py [--outdir DIR] [--max-new N]   -> logs/partB_refine.log, results/partB_refine.json

Round 7 Part 0 (no numerical change): every sigma/48 evaluation is appended (and fsync'ed) to the incremental checkpoint
results/partB_refine.ckpt.jsonl as it completes (job index, job, record; not committed); a re-run resumes and evaluates only
the missing jobs. --outdir redirects the checkpoint, partB_refine.json and the log to DIR (verification runs); --max-new N
stops after N newly completed evaluations (an interruption, to exercise the resume).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.dirname(HERE)
R4 = os.path.join(EV, 'round4'); R5 = os.path.join(EV, 'round5')
sys.path.insert(0, R4); sys.path.insert(0, R5); sys.path.insert(0, HERE)
import numpy as np
import robustness as rb                                   # round 5, unchanged
import robustness_margin as rm
ew, ef, dr = rb.ew, rb.ef, rb.dr

s, K, B, D = rb.s, rb.K, rb.B, rb.D
LOGS, RES = rm.LOGS, rm.RES
N48 = 48; TOL = 0.05
NAME = 'U'
I24 = rb.INP[NAME]
_z48, _dz48 = dr.make_grid(I24['xbar'], N48, 8*D, 'B')
I48 = dict(z=_z48, dz=_dz48, G=len(_z48))
VERDICT = '离散方案下'
OUT = dict(ckpt=os.path.join(RES, 'partB_refine.ckpt.jsonl'), json=os.path.join(RES, 'partB_refine.json'))


def eval_table48(tab, xth):
    """the frozen coarse table tab (round-5 LookupTable) moved to the sigma/48 grid (FineTable) at the true output xth."""
    z, dz, G = I48['z'], I48['dz'], I48['G']; ft = ew.FineTable(tab)
    a, b = ew.windows88(xth, z[0], dz, G)
    assert int((a-ft.X).min()) >= 0 and int((b+ft.X).max()) <= G-1, "a window would cross the frozen fine grid"
    def run(xk, n_, margin, version):
        assert xk is xth and n_ == N48 and version == 'B'
        return ew.forward(z, dz, s, xth, a, b, ft)
    with rb.patched(dr, input_arrays=lambda nm: (xth, I24['H']), run_ab=run):
        rec, J, r = dr.evaluate('B', NAME, N48, 8)
    rec['path'] = 'FineTable on the frozen sigma/48 grid, general window path (R = 88 sigma), windows centred at x(theta)'
    return rec, J, r

def eval_sod48(xth):
    mg = dr.make_grid
    with rb.patched(dr, input_arrays=lambda nm: (xth, I24['H']), make_grid=lambda xk, n, margin, version: mg(I24['xbar'], n, margin, version)):
        rec, J, r = ef.evaluate_fast(NAME, N48)
    rec['path'] = 'accepted operator (evaluator_fast.evaluate_fast, n = 48), frozen nominal fine grid, windows centred at x(theta)'
    return rec, J, r

_T = None; _JOBS = None
def _init(): ew._warmup()

def _job(i):
    kind, t, theta = _JOBS[i]
    xth = rb.x_of(NAME, np.asarray(theta)); t0 = time.perf_counter()
    rec, J, r = eval_sod48(xth) if kind == 'SoD' else eval_table48(_T[f'{NAME}/{t}'], xth)
    keep = {k: (rec[k].item() if hasattr(rec[k], 'item') else rec[k]) for k in rb.KEEP if k in rec}
    keep['assert_fail'] = int((~r['assert_ok']).sum()); keep['wall_s'] = time.perf_counter()-t0; keep['G'] = int(rec['G'])
    return i, keep

def main(out, max_new=None):
    global _T, _JOBS
    import multiprocessing as mp
    S_ = json.load(open(rm.SUMMARY)); sel = S_['selected']
    out(f"B0 selected table {sel} ({S_['selected_label']}); fine grid G = {I48['G']}, h = sigma/48 = {I48['dz']!r}; coarse G = {I24['G']}")
    # nested-grid check of the nearest-node rule (round 4 DECISION C-3) on the frozen grids
    zc, zf = I24['z'], I48['z']; n2 = min(len(zf[0::2]), len(zc)); even_ok = bool(np.array_equal(zf[0::2][:n2], zc[:n2]))
    jo = np.arange(1, len(zf), 2); ic = jo//2; m_ = ic+1 < len(zc); jo, ic = jo[m_], ic[m_]
    tie_dev = float(np.max(np.abs(np.abs(zf[jo]-zc[ic])-np.abs(zc[ic+1]-zf[jo]))))
    out(f"B0 nested grids: fine node 2i == coarse node i bitwise {even_ok}; odd fine nodes equidistant to within {tie_dev!r} (ties -> smaller index, j // 2)")
    A = rm.read_archive(); q = [r for r in A if r['table'] == sel]
    nom = [r for r in q if r['cls'] == 'theta0'][0]; p80 = [r for r in q if r['cls'] != 'theta0']; p26 = [r for r in p80 if r['selection26']]
    pts = [('theta_0', nom, True), ('max E[N] in the 26', max(p26, key=lambda r: r['EN']), True),
           ('min Delta m_D in the 26', min(p26, key=lambda r: r['dmD']), True),
           ('max E[N] in the 80 (supplementary)', max(p80, key=lambda r: r['EN']), False),
           ('min Delta m_D in the 80 (supplementary)', min(p80, key=lambda r: r['dmD']), False)]
    uniq = {}
    for role, r, main_ in pts: uniq.setdefault(rm.point_id(r), (r, []))[1].append((role, main_))
    _T = rm.load_tables()
    _JOBS = []
    for pid, (r, roles) in uniq.items():
        _JOBS += [('SoD', None, r['theta']), ('table', sel, r['theta'])]
    _JOBS += [('SoD', None, list(rb.THETA0)), ('table', 'm000', list(rb.THETA0))] if sel != 'm000' else []
    # per-evaluation incremental checkpoint (round 7 Part 0)
    jkey = [json.dumps([k_, t_, [float(v) for v in th]]) for k_, t_, th in _JOBS]
    res = {}
    if os.path.exists(OUT['ckpt']):
        for l in open(OUT['ckpt']):
            c = json.loads(l); assert c['job'] == jkey[c['i']], "checkpoint does not belong to this job list"; res[c['i']] = c['rec']
    todo = [i for i in range(len(_JOBS)) if i not in res]
    out(f"B0-CKPT {OUT['ckpt']}: {len(res)} evaluations resumed, {len(todo)} to evaluate")
    t0 = time.perf_counter(); n_new = 0
    if todo:
        with mp.get_context('fork').Pool(4, initializer=_init) as pool, open(OUT['ckpt'], 'a') as fc:
            for i, keep in pool.imap_unordered(_job, todo, chunksize=1):
                fc.write(json.dumps(dict(i=i, job=jkey[i], rec=keep), ensure_ascii=False)+'\n'); fc.flush(); os.fsync(fc.fileno())
                res[i] = keep; n_new += 1
                if max_new is not None and n_new >= max_new:
                    pool.terminate(); break
    wall = time.perf_counter()-t0
    if len(res) < len(_JOBS):
        out(f"B0-CKPT stopped after {n_new} new evaluations ({len(res)}/{len(_JOBS)} in the checkpoint; --max-new {max_new}); re-run to resume")
        return None
    out(f"B0 {len(_JOBS)} sigma/48 evaluations in {wall:.1f} s with 4 workers; max |sum m - 1| {max(v['mass_dev_max'] for v in res.values())!r}; "
        f"min m {min(v['min_m'] for v in res.values())!r}; assertion failures {sum(v['assert_fail'] for v in res.values())}")
    # path regression at theta_0 (m = 0 table and SoD vs round 4 C2)
    c2 = json.load(open(os.path.join(R4, 'results', 'c2.json')))
    ji = len(_JOBS)-2 if sel != 'm000' else [i for i, j in enumerate(_JOBS) if j[1] == 'm000'][0]-1
    rs0, rt0 = res[ji], res[ji+1]
    reg = dict(SoD_EN=rs0['EN'] == c2['U/SoD']['EN48'], SoD_logdet=rs0['logdet'] == c2['U/SoD']['logdet48'], SoD_S=rs0['S'] == c2['U/SoD']['S48'],
               m000_EN=rt0['EN'] == c2['U/W']['EN48'], m000_logdet=rt0['logdet'] == c2['U/W']['logdet48'], m000_S=rt0['S'] == c2['U/W']['S48'])
    out(f"B-REG theta_0 through the mismatch sigma/48 path vs round4/results/c2.json (exact): {reg}; SoD E[N]_48 {rs0['EN']!r} (c2 {c2['U/SoD']['EN48']!r}), "
        f"m000 E[N]_48 {rt0['EN']!r} (c2 {c2['U/W']['EN48']!r}), m000 log det_48 {rt0['logdet']!r} (c2 {c2['U/W']['logdet48']!r})")
    rows = []; k = 0
    for pid, (r, roles) in uniq.items():
        so, ta = res[k], res[k+1]; k += 2
        N24, N48_ = r['EN'], ta['EN']; dm24 = r['dmD']; dm48 = ta['logdet']-so['logdet']
        v24, v48 = bool(N24 <= B), bool(N48_ <= B)
        reasons = []
        if v24 != v48: reasons.append('预算判定翻转')
        if abs(N48_-N24) > TOL: reasons.append(f'|N_48 - N_24| = {abs(N48_-N24):.4f} > {TOL}')
        if min(abs(N24-B), abs(N48_-B)) <= TOL: reasons.append(f'贴预算线 min(|N_24 - B|, |N_48 - B|) = {min(abs(N24-B), abs(N48_-B)):.4f} <= {TOL}')
        is_main = any(m_ for _, m_ in roles)
        d = dict(point=pid, roles=[x for x, _ in roles], main=is_main, theta=r['theta'], eps_attained=r['eps_attained'],
                 EN24=N24, EN48=N48_, dEN=N48_-N24, dmD24=dm24, dmD48=dm48, ddmD=dm48-dm24, logdet24=r['logdet'], logdet48=ta['logdet'],
                 logdet_SoD24=r['logdet_SoD'], logdet_SoD48=so['logdet'], EN_SoD24=r['EN_SoD'], EN_SoD48=so['EN'],
                 verdict24=('E[N] <= B' if v24 else 'E[N] > B'), verdict48=('E[N] <= B' if v48 else 'E[N] > B'),
                 slack24=B-N24, slack48=B-N48_, flip=v24 != v48, undetermined=bool(reasons), reasons=reasons,
                 label=('精度未定' if reasons else '未标精度未定') if is_main else ('（补充，不参与判定）'+('精度未定' if reasons else '未标精度未定')),
                 assert_fail=ta['assert_fail']+so['assert_fail'], mass_dev_max=ta['mass_dev_max'], min_m=ta['min_m'],
                 boundary_steps48=ta.get('boundary_steps'), wall_s=ta['wall_s']+so['wall_s'])
        rows.append(d)
        for kk, v in d.items():
            if kk != 'theta': out(f"B2 {pid} {kk} {v!r}")
    out(f"B2-TAB {'point':28s} {'role':38s} {'E[N]_24':>10s} {'E[N]_48':>10s} {'dE[N]':>8s} {'dmD_24':>8s} {'dmD_48':>8s} {'ddmD':>8s} "
        f"{'v24':>9s} {'v48':>9s}  label")
    for d in rows:
        out(f"B2-TAB {d['point']:28s} {'; '.join(d['roles'])[:38]:38s} {d['EN24']:10.4f} {d['EN48']:10.4f} {d['dEN']:+8.4f} {d['dmD24']:8.4f} "
            f"{d['dmD48']:8.4f} {d['ddmD']:+8.4f} {d['verdict24']:>9s} {d['verdict48']:>9s}  {d['label']} {d['reasons']}")
    main_und = [d['point'] for d in rows if d['main'] and d['undetermined']]
    out(f"B3 main check points 精度未定: {main_und} ({len(main_und)} of {sum(d['main'] for d in rows)}); verdict item (b) "
        f"{'met' if not main_und else 'not met'} ({VERDICT})")
    json.dump(dict(selected=sel, selected_label=S_['selected_label'], rows=rows, regression=reg, main_undetermined=main_und, wall_s=wall,
                   n_evals=len(_JOBS), G48=I48['G'], nested_even=even_ok, tie_dev=tie_dev), open(OUT['json'], 'w'),
              indent=1, ensure_ascii=False)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--outdir'); ap.add_argument('--max-new', type=int); A_ = ap.parse_args()
    if A_.outdir:
        os.makedirs(A_.outdir, exist_ok=True)
        OUT.update(ckpt=os.path.join(A_.outdir, 'partB_refine.ckpt.jsonl'), json=os.path.join(A_.outdir, 'partB_refine.json'))
    sys.stdout = dr.Tee(os.path.join(A_.outdir or LOGS, 'partB_refine.log'), 'a' if A_.outdir else 'w')
    print(f"=== refine_check.py started {time.strftime('%Y-%m-%d %H:%M:%S')}; spec reports_CN/plan_round6_CN.md v3.2 §2; B = {B!r}; tolerance {TOL}")
    t0 = time.perf_counter(); main(print, A_.max_new); print(f"=== refine_check.py done in {time.perf_counter()-t0:.1f} s")
